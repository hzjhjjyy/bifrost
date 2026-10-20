#!/usr/bin/env python3
"""Loopback echo upstream for the OpenAI passthrough path-mode harness.

Run: python3 tests/e2e/api/fixtures/path-mode.py --port 18082
Configure path-test as an OpenAI-based custom provider with this loopback base URL,
max_retries=0, and a synthetic key. Select the harness feature 'passthrough path
mode' with INCLUDE_PREVIEW=1. No real provider account is needed.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class PathModeFixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def handle_request(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
        try:
            request = json.loads(body) if body else {}
        except json.JSONDecodeError:
            request = {}
        record = {"path": self.path, "method": self.command, "body": body,
                  "control_header": self.headers.get("x-bf-passthrough-path-mode")}
        stream = isinstance(request, dict) and request.get("stream") is True
        content_type = "text/event-stream" if stream else "application/json"
        payload = json.dumps(record, separators=(",", ":")).encode()
        if self.path == "/v1/models" and self.command == "GET":
            payload = b'{"object":"list","data":[{"id":"test-model","object":"model","owned_by":"fixture"}]}'
        elif stream:
            payload = b"data: " + payload + b"\n\ndata: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = handle_request
    do_POST = handle_request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), PathModeFixture) as server:
        print(f"Path-mode fixture listening on 127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
