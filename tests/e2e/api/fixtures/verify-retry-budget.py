#!/usr/bin/env python3
"""Verify a built gateway against loopback-only upstreams; no provider keys needed.

Usage: python3 tests/e2e/api/fixtures/verify-retry-budget.py /path/to/bifrost-http
The binary must be built with local core sources (go.work or Dockerfile.local).
"""

import argparse
import importlib.util
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("retry_fixture", Path(__file__).with_name("retry-disconnect.py"))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    upstream = fixture.ThreadingHTTPServer(("127.0.0.1", 0), fixture.DisconnectFixture)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()

    def request(url, method="GET", body=None, headers=None):
        parsed = urlsplit(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=15)
        try:
            # urllib forces Connection: close, which passthrough forwards and would
            # defeat the pooled-connection regression this check must exercise.
            conn.request(method, parsed.path + ("?" + parsed.query if parsed.query else ""), body, headers or {})
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    try:
        with tempfile.TemporaryDirectory(prefix="bifrost-retry-") as directory:
            app = Path(directory)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            upstream_url = f"http://127.0.0.1:{upstream.server_port}"
            providers = {}
            for name, retries in (("retry-zero", 0), ("retry-one", 1), ("retry-two", 2), ("retry-default", None)):
                network = {"base_url": upstream_url, "retry_backoff_initial": 100, "retry_backoff_max": 100, "max_conns_per_host": 1}
                if retries is not None:
                    network["max_retries"] = retries
                providers[name] = {
                    "keys": [{"name": "fixture", "value": "fixture-only", "models": ["test-model"], "weight": 1}],
                    "network_config": network,
                    "custom_provider_config": {"base_provider_type": "openai"},
                }
            (app / "catalog.json").write_text("{}")
            config = {
                "client": {"enable_logging": False, "enforce_auth_on_inference": False},
                "providers": providers,
                "framework": {"pricing": {"pricing_url": (app / "catalog.json").as_uri(),
                            "model_parameters_url": (app / "catalog.json").as_uri(),
                            "mcp_library_sync_interval": 0, "live_models_sync_interval": 0}},
            }
            (app / "config.json").write_text(json.dumps(config))
            # Do not inherit real provider credentials or proxy settings.
            env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ}
            with (app / "gateway.log").open("w+") as log:
                gateway = subprocess.Popen([str(binary), "-host", "127.0.0.1", "-port", str(port),
                                            "-app-dir", str(app), "-log-level", "error"],
                                           stdout=log, stderr=log, env=env)
                try:
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        if gateway.poll() is not None:
                            raise AssertionError(f"gateway exited {gateway.returncode}")
                        try:
                            if request(base + "/health")[0] == 200:
                                break
                        except (http.client.HTTPException, OSError):
                            pass
                        time.sleep(0.2)
                    else:
                        raise AssertionError("gateway did not become healthy")
                    checks = 0
                    for name, retries in (("retry-zero", 0), ("retry-one", 1), ("retry-two", 2), ("retry-default", 0)):
                        for method in ("POST", "GET"):
                            headers = {"x-model-provider": name, "Content-Type": "application/json"}
                            status, body = request(base + "/openai_passthrough/warm", headers=headers)
                            assert status == 200, (status, body)
                            warm_port = json.loads(body)["client_port"]
                            rid = f"{name}-{method}"
                            path = "/chat/completions" if method == "POST" else "/models"
                            payload = b'{"model":"test-model","messages":[{"role":"user","content":"disconnect"}]}' if method == "POST" else None
                            status, body = request(base + "/openai_passthrough" + path + "?id=" + rid, method, payload, headers)
                            assert status == 502, (status, body)
                            assert json.loads(body)["error"]["type"] == "provider_connection_failed", body
                            records = fixture.DisconnectFixture.records[rid]
                            # One pooled-connection replay plus the core retry budget.
                            want = 1 if retries == 0 else retries + 2
                            assert len(records) == want, (rid, len(records), want)
                            assert records[0]["client_port"] == warm_port, (rid, "connection not reused")
                            assert all(r["method"] == method and r["body"] == (payload or b"").decode() for r in records), records
                            print(f"PASS {rid}: reused connection, {want} upstream request(s)")
                            checks += 1
                        for fault in ("429", "503", "disconnect", "recover"):
                            rid = f"{name}-{fault}"
                            headers = {"Content-Type": "application/json", "x-bf-eh-x-retry-test-id": rid,
                                       "x-bf-eh-x-retry-test-fault": fault}
                            payload = json.dumps({"model": name + "/test-model", "messages": [{"role": "user", "content": "retry"}]}).encode()
                            status, body = request(base + "/v1/chat/completions", "POST", payload, headers)
                            recovered = fault == "recover" and retries > 0
                            want_status = 200 if recovered else (429 if fault == "429" else (502 if fault == "disconnect" else 503))
                            assert status == want_status, (rid, status, body)
                            if recovered:
                                assert json.loads(body)["choices"][0]["message"]["content"] == "ok", body
                            want = 2 if recovered else retries + 1
                            records = fixture.DisconnectFixture.records.get(rid, [])
                            assert len(records) == want, (rid, len(records), want)
                            print(f"PASS {rid}: HTTP {status}, {want} upstream request(s)")
                            checks += 1
                    print(f"PASS: {checks} compiled-gateway retry checks")
                except Exception:
                    log.flush()
                    log.seek(0)
                    print(log.read())
                    raise
                finally:
                    gateway.terminate()
                    try:
                        gateway.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        gateway.kill()
                        gateway.wait()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
