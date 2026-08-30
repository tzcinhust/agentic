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

import hashlib
import json
import os
import random
import socketserver
import sys
import threading
import time
from collections import deque
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx

UPSTREAM = os.environ.get("SHIM_UPSTREAM", "").rstrip("/")
PORT = int(os.environ.get("SHIM_PORT", "8765"))
PREFIX = os.environ.get("SHIM_PREFIX", "/openai/v1")
ATTEMPTS = int(os.environ.get("SHIM_ATTEMPTS", "5"))
TIMEOUT = float(os.environ.get("SHIM_TIMEOUT", "600"))
VERBOSE = os.environ.get("SHIM_VERBOSE", "1") == "1"
RPM = int(os.environ.get("SHIM_RPM", "45"))
BURST = int(os.environ.get("SHIM_BURST", "5"))
BURST_WINDOW = float(os.environ.get("SHIM_BURST_WINDOW", "1.0"))
RETRY_STATUSES = {429, 502, 503, 504}
LEDGER_PATH = Path(os.environ["SHIM_LEDGER_PATH"]).resolve() if os.environ.get("SHIM_LEDGER_PATH") else None

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
_ledger_lock = threading.Lock()


def _provider_origin_sha256() -> str:
    parsed = urlsplit(UPSTREAM)
    port = parsed.port
    default_port = (parsed.scheme == "https" and port == 443) or (parsed.scheme == "http" and port == 80)
    origin = f"{parsed.scheme.lower()}://{(parsed.hostname or '').lower()}"
    if port and not default_port:
        origin += f":{port}"
    return hashlib.sha256(origin.encode("utf-8")).hexdigest()


def _request_route(path: str) -> str:
    if path.startswith("/v1/chat/completions"):
        return "agent_chat_completions"
    if path.startswith("/openai/v1/responses"):
        return "official_eval_responses"
    return "other"


def _usage_summary(payload: bytes) -> dict[str, int] | None:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("usage"), dict):
        return None
    usage = value["usage"]
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    input_details = usage.get("input_tokens_details", usage.get("prompt_tokens_details", {}))
    output_details = usage.get("output_tokens_details", usage.get("completion_tokens_details", {}))
    cached = input_details.get("cached_tokens", 0) if isinstance(input_details, dict) else 0
    reasoning = output_details.get("reasoning_tokens", 0) if isinstance(output_details, dict) else 0
    total = usage.get("total_tokens", input_tokens + output_tokens)
    if not all(isinstance(item, int) and item >= 0 for item in (input_tokens, output_tokens, cached, reasoning, total)):
        return None
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning,
        "total_tokens": total,
    }


def _append_ledger(record: dict[str, object]) -> None:
    if LEDGER_PATH is None:
        return
    encoded = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    with _ledger_lock:
        with LEDGER_PATH.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()


class SlidingWindowLimiter:
    """Thread-safe shared limiter for agent, simulator, and judge traffic."""

    def __init__(self, *, rpm: int, burst: int, burst_window: float):
        if rpm < 1 or burst < 1 or burst_window <= 0:
            raise ValueError("SHIM_RPM, SHIM_BURST, and SHIM_BURST_WINDOW must be positive")
        self.rpm = rpm
        self.burst = burst
        self.burst_window = burst_window
        self._minute: deque[float] = deque()
        self._burst: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            with self._lock:
                while self._minute and now - self._minute[0] >= 60.0:
                    self._minute.popleft()
                while self._burst and now - self._burst[0] >= self.burst_window:
                    self._burst.popleft()
                waits = []
                if len(self._minute) >= self.rpm:
                    waits.append(60.0 - (now - self._minute[0]))
                if len(self._burst) >= self.burst:
                    waits.append(self.burst_window - (now - self._burst[0]))
                if not waits:
                    self._minute.append(now)
                    self._burst.append(now)
                    return
                wait_for = max(0.01, max(waits))
            time.sleep(wait_for)


LIMITER = SlidingWindowLimiter(rpm=RPM, burst=BURST, burst_window=BURST_WINDOW)


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
        # The locked evaluator calls /openai/v1/* while the custom agent calls
        # /v1/*.  Routing both through this process makes the RPM ceiling truly
        # global instead of one limit per client.
        for prefix in dict.fromkeys((PREFIX, "/openai/v1", "/v1")):
            if path.startswith(prefix):
                return UPSTREAM + path[len(prefix) :]
        return None

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
            attempt_started = time.time()
            try:
                LIMITER.acquire()
                response = CLIENT.request(self.command, target, headers=headers, content=body)
                payload = response.content
                _append_ledger(
                    {
                        "schema_version": "1.0.0",
                        "event": "upstream_response",
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "request_id": request_id,
                        "attempt": attempt,
                        "route": _request_route(self.path),
                        "method": self.command,
                        "audit_id": self.headers.get("X-PWM-Audit-ID"),
                        "task_key": self.headers.get("X-PWM-Task-Key"),
                        "status_code": response.status_code,
                        "retryable_status": response.status_code in RETRY_STATUSES,
                        "elapsed_ms": round((time.time() - attempt_started) * 1000),
                        "usage": _usage_summary(payload),
                    }
                )
                if response.status_code in RETRY_STATUSES and attempt < ATTEMPTS:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        delay = 0.0
                    delay = max(delay, min(1.5 * (2 ** (attempt - 1)), 30.0))
                    delay += random.uniform(0.0, min(1.0, delay * 0.2))
                    if VERBOSE:
                        print(
                            f"[{request_id}] HTTP {response.status_code} retry "
                            f"{attempt}/{ATTEMPTS - 1} in {delay:.1f}s",
                            flush=True,
                        )
                    time.sleep(delay)
                    continue
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
            except httpx.TransportError as error:
                last_error = f"{type(error).__name__}: {error}"
                _append_ledger(
                    {
                        "schema_version": "1.0.0",
                        "event": "transport_error",
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "request_id": request_id,
                        "attempt": attempt,
                        "route": _request_route(self.path),
                        "method": self.command,
                        "audit_id": self.headers.get("X-PWM-Audit-ID"),
                        "task_key": self.headers.get("X-PWM-Task-Key"),
                        "status_code": None,
                        "retryable_status": True,
                        "elapsed_ms": round((time.time() - attempt_started) * 1000),
                        "error_type": type(error).__name__,
                        "usage": None,
                    }
                )
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
    if not UPSTREAM:
        sys.exit("Set SHIM_UPSTREAM, e.g. https://ai.novacode.top/v1")

    class Server(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    if LEDGER_PATH is not None:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with LEDGER_PATH.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            sys.exit(f"Refusing to overwrite relay ledger: {LEDGER_PATH}")
        _append_ledger(
            {
                "schema_version": "1.0.0",
                "event": "session_start",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "provider": "novacode",
                "upstream_origin_sha256": _provider_origin_sha256(),
                "rpm": RPM,
                "burst": BURST,
                "burst_window_seconds": BURST_WINDOW,
                "attempts": ATTEMPTS,
            }
        )

    server = Server(("127.0.0.1", PORT), Handler)
    print(f"eval shim listening on http://127.0.0.1:{PORT}{PREFIX}/  ->  {UPSTREAM}/", flush=True)
    print(f"  set STATE_BENCH_EVAL_ENDPOINT=\"http://127.0.0.1:{PORT}\"", flush=True)
    print(f"  set STATE_BENCH_AGENT_BASE_URL=\"http://127.0.0.1:{PORT}/v1\"", flush=True)
    print(
        f"  retries={ATTEMPTS} timeout={TIMEOUT}s rpm={RPM} burst={BURST}/{BURST_WINDOW:g}s",
        flush=True,
    )
    if LEDGER_PATH is not None:
        print(f"  append-only usage ledger={LEDGER_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshim stopped", flush=True)


if __name__ == "__main__":
    main()
