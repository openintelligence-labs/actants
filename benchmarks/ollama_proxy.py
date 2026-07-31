"""A transparent recording proxy in front of Ollama.

Every framework in this benchmark is pointed at this proxy instead of Ollama
directly. The proxy forwards requests verbatim and records, per request:

* ``wire_seconds`` — time from the proxy sending the request to Ollama until
  the last byte of Ollama's response arrives. This is the model's work plus
  loopback transport, and is identical work for every framework.
* ``server_total_ns`` / ``eval_count`` — Ollama's own reported timings, when
  present in the response body.

The runner subtracts the summed ``wire_seconds`` for a task from that task's
end-to-end wall time. The remainder is the framework's own overhead: request
construction, schema serialisation, response parsing, agent-loop bookkeeping.
This is the only way to compare frameworks fairly when model time (hundreds of
milliseconds) dwarfs framework time (single-digit milliseconds).

The proxy is deliberately dependency-free (stdlib only) so it can run in any
of the comparison venvs.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://localhost:11434"

_records: list[dict] = []
_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # noqa: A002 - silence access log
        pass

    def _control(self) -> bool:
        """Handle the proxy's own control endpoints. Returns True if handled."""
        if self.path == "/__bench__/records":
            with _lock:
                body = json.dumps(_records).encode()
            self._respond(200, body, "application/json")
            return True
        if self.path == "/__bench__/reset":
            with _lock:
                _records.clear()
            self._respond(200, b"{}", "application/json")
            return True
        return False

    def _respond(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self._control():
            return
        self._forward(b"")

    def do_POST(self) -> None:  # noqa: N802
        if self._control():
            return
        length = int(self.headers.get("Content-Length") or 0)
        self._forward(self.rfile.read(length) if length else b"")

    def _forward(self, body: bytes) -> None:
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "connection", "accept-encoding")
        }
        request = urllib.request.Request(
            UPSTREAM + self.path,
            data=body if body else None,
            headers=headers,
            method=self.command,
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                payload = response.read()
                status = response.status
                ctype = response.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            status = exc.code
            ctype = exc.headers.get("Content-Type", "application/json")
        wire = time.perf_counter() - start

        record: dict = {"path": self.path, "wire_seconds": wire, "status": status}
        _annotate(record, payload)
        with _lock:
            _records.append(record)

        self._respond(status, payload, ctype)


def _annotate(record: dict, payload: bytes) -> None:
    """Pull Ollama's self-reported timings out of the response, if present.

    Handles both single JSON objects and newline-delimited streaming bodies
    (in which case the final object carries the totals).
    """
    text = payload.decode("utf-8", "replace").strip()
    if not text:
        return
    obj = None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        # Server-sent-events framing used by the OpenAI-compatible endpoints.
        if line.startswith("data: "):
            line = line[6:].strip()
        if line == "[DONE]":
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            obj = candidate
            break
    if not isinstance(obj, dict):
        return
    for key in ("total_duration", "load_duration", "eval_count", "prompt_eval_count"):
        if key in obj:
            record[key] = obj[key]
    usage = obj.get("usage")
    if isinstance(usage, dict):
        record["usage"] = usage


def serve(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11500
    serve(port)
    print(f"proxy listening on 127.0.0.1:{port} -> {UPSTREAM}", flush=True)
    threading.Event().wait()
