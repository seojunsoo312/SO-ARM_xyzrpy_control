"""Local JSON pose feed for hand-eye capture. Motion owns the bus; clients only GET."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen

_DISCONNECT = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

POSE_HOST = "127.0.0.1"
POSE_PORT = 8765
POSE_PATH = "/pose"
POSE_URL = f"http://{POSE_HOST}:{POSE_PORT}{POSE_PATH}"


def fetch_pose(*, timeout_s: float = 1.0) -> dict | None:
    """None if the pendant process is not serving."""
    try:
        with urlopen(POSE_URL, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _make_handler(payload_fn: Callable[[], dict]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != POSE_PATH:
                self.send_error(404)
                return
            try:
                payload = payload_fn()
                body = json.dumps(payload).encode("utf-8")
                code = 200
            except Exception as exc:
                body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
                code = 500
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except _DISCONNECT:
                return

        def handle(self) -> None:
            try:
                super().handle()
            except _DISCONNECT:
                return

        def finish(self) -> None:
            try:
                super().finish()
            except _DISCONNECT:
                return

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


class PoseServer:
    def __init__(
        self,
        payload_fn: Callable[[], dict],
        *,
        host: str = POSE_HOST,
        port: int = POSE_PORT,
    ) -> None:
        self._payload_fn = payload_fn
        self._host = host
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        class _Server(ThreadingHTTPServer):
            allow_reuse_address = True

            def handle_error(self, request, client_address) -> None:
                err = sys.exc_info()[1]
                if isinstance(err, _DISCONNECT):
                    return
                super().handle_error(request, client_address)

        handler = _make_handler(self._payload_fn)
        self._httpd = _Server((self._host, self._port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)
