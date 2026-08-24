"""Optional standard-library HTTP adapter for ApiService."""

from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .service import ApiService


class ApiHTTPRequestHandler(BaseHTTPRequestHandler):
    """Translate HTTP requests to the transport-neutral ApiService facade."""

    service: ApiService | None = None

    def _dispatch(self) -> None:
        if self.service is None:
            self.send_error(500, "API service is not configured")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else None
        response = asyncio.run(self.service.handle(self.command, urlsplit(self.path).path, raw_body))
        encoded = b"" if response.body is None else json.dumps(response.body).encode("utf-8")
        self.send_response(response.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if encoded:
            self.wfile.write(encoded)

    do_GET = _dispatch
    do_POST = _dispatch
    do_DELETE = _dispatch

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_http_server(host: str, port: int, service: ApiService) -> ThreadingHTTPServer:
    """Create, but do not start, a standard-library HTTP server."""

    handler = type("ConfiguredApiHTTPRequestHandler", (ApiHTTPRequestHandler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)
