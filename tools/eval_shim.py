"""Azure-shape reverse proxy so STATE-Bench's locked eval client can target a relay.

STATE-Bench builds its eval base URL as:

    _azure_openai_v1_base_url(endpoint) == endpoint.rstrip("/") + "/openai/v1/"

so a relay served at ``https://host/v1`` cannot be plugged into
``STATE_BENCH_EVAL_ENDPOINT`` directly. This shim listens on localhost and
rewrites ``/openai/v1/<rest>`` -> ``<UPSTREAM>/<rest>``, forwarding headers and
body untouched. Nothing in the STATE-Bench checkout needs to be edited, so the
protocol prompt hashes stay intact.

Run:
    SHIM_UPSTREAM="https://ai.novacode.top/v1" SHIM_PORT=8765 python tools/eval_shim.py

Then in STATE-Bench's .env:
    STATE_BENCH_EVAL_ENDPOINT="http://127.0.0.1:8765"
    STATE_BENCH_EVAL_DEPLOYMENTS="gpt-5.4"
    STATE_BENCH_EVAL_API_KEY="<relay key>"

Retries are built in because local CONNECT proxies (Clash/v2ray) drop tunnels
intermittently, which otherwise surfaces as SSL UNEXPECTED_EOF mid-run.
"""

from __future__ import annotations

import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

UPSTREAM = os.environ.get("SHIM_UPSTREAM", "").rstrip("/")
PORT = int(os.environ.get("SHIM_PORT", "8765"))
PREFIX = os.environ.get("SHIM_PREFIX", "/openai/v1")
ATTEMPTS = int(os.environ.get("SHIM_ATTEMPTS", "5"))
TIMEOUT = float(os.environ.get("SHIM_TIMEOUT", "600"))
VERBOSE = os.environ.get("SHIM_VERBOSE", "1") == "1"

if not UPSTREAM:
    sys.exit("Set SHIM_UPSTREAM, e.g. https://ai.novacode.top/v1")

CLIENT = httpx.Client(
    follow_redirects=True,
    timeout=httpx.Timeout(TIMEOUT),
    trust_env=True,
)
_HOP_BY_HOP = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "accept-encoding",
}
_counter = 0
_counter_lock = threading.Lock()


def _next_id() -> int:
    global _counter
    with _counter_lock:
        _counter += 1
        return _counter


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "state-bench-eval-shim"

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003 - silence default logging
        pass

    def _target(self) -> str | None:
        path = self.path
        if not path.startswith(PREFIX):
            return None
        return UPSTREAM + path[len(PREFIX) :]

    def _forward(self) -> None:
        target = self._target()
        request_id = _next_id()
        if target is None:
            self._respond(404, b'{"error":{"message":"shim: path outside prefix"}}')
            if VERBOSE:
                print(f"[{request_id}] 404 unmapped {self.path}", flush=True)
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP
        }

        started = time.time()
        last_error = "unknown"
        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = CLIENT.request(self.command, target, headers=headers, content=body)
                payload = response.content
                self._respond(
                    response.status_code,
                    payload,
                    response.headers.get("Content-Type", "application/json"),
                )
                if VERBOSE:
                    elapsed = time.time() - started
                    note = f" (retried x{attempt - 1})" if attempt > 1 else ""
                    print(
                        f"[{request_id}] {response.status_code} {self.command} {self.path} "
                        f"{elapsed:.1f}s {len(payload)}B{note}",
                        flush=True,
                    )
                return
            except Exception as error:  # noqa: BLE001 - transport flake, retry
                last_error = f"{type(error).__name__}: {error}"
                if attempt < ATTEMPTS:
                    if VERBOSE:
                        print(f"[{request_id}] transport retry {attempt}/{ATTEMPTS - 1}: {last_error}", flush=True)
                    time.sleep(min(1.5 * attempt, 8.0))

        self._respond(502, f'{{"error":{{"message":"shim upstream failed: {last_error}"}}}}'.encode())
        if VERBOSE:
            print(f"[{request_id}] 502 {self.command} {self.path} gave up: {last_error}", flush=True)

    def _respond(self, status: int, payload: bytes, content_type: str = "application/json") -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    do_POST = _forward
    do_GET = _forward
    do_DELETE = _forward
    do_PATCH = _forward
    do_PUT = _forward


def main() -> None:
    class Server(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server(("127.0.0.1", PORT), Handler)
    print(f"eval shim listening on http://127.0.0.1:{PORT}{PREFIX}/  ->  {UPSTREAM}/", flush=True)
    print(f"  set STATE_BENCH_EVAL_ENDPOINT=\"http://127.0.0.1:{PORT}\"", flush=True)
    print(f"  retries={ATTEMPTS} timeout={TIMEOUT}s", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshim stopped", flush=True)


if __name__ == "__main__":
    main()
