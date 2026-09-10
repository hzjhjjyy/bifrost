#!/usr/bin/env python3
"""Loopback upstream for provider-harness's zero-retry regression.

Run: python3 tests/e2e/api/fixtures/retry-disconnect.py --port 18081
Add this provider to a local gateway's config.json (all credentials synthetic):
  "providers": {"retry-test": {
    "keys": [{"name": "fixture", "value": "fixture-only", "models": ["test-model"], "weight": 1}],
    "network_config": {"base_url": "http://127.0.0.1:18081", "max_retries": 0},
    "custom_provider_config": {"base_provider_type": "openai", "allowed_requests": {"passthrough": true}}
  }}
No real provider account is needed.

Filter the harness with --provider openai --feature zero-retry, then run Newman
against a locally configured gateway with these environment variables:
  baseUrl=http://127.0.0.1:8080
  retryFixtureUrl=http://127.0.0.1:18081
  include_preview=1
The [PREVIEW] cases require this explicit fixture setup and are excluded from
ordinary paid-provider sweeps. If the gateway requires a VK, supply a synthetic
authorized VK via the retryFixtureVK environment variable.
"""

import argparse
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


class DisconnectFixture(BaseHTTPRequestHandler):
    records = {}
    lock = threading.Lock()

    def log_message(self, *_):
        pass

    def handle_request(self):
        url = urlsplit(self.path)
        request_id = parse_qs(url.query).get("id", [""])[0]
        if url.path == "/records" and self.command == "GET":
            with self.lock:
                payload = json.dumps(self.records.get(request_id, [])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if not request_id or url.path not in ("/v1/chat/completions", "/v1/user/balance"):
            self.send_error(400, "expected a regression path and unique id")
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        with self.lock:
            self.records.setdefault(request_id, []).append({
                "method": self.command,
                "path": url.path,
                "body": body.decode(),
                "complete_body_bytes": len(body),
                "client_port": self.client_address[1],
            })
        self.close_connection = True
        self.connection.shutdown(socket.SHUT_RDWR)

    do_GET = handle_request
    do_POST = handle_request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), DisconnectFixture) as server:
        print(f"Disconnect fixture listening on 127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
