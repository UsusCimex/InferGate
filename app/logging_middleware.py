from __future__ import annotations

import logging
import time

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("infergate.access")

_SKIP_PATHS = frozenset({"/health", "/openapi.json"})


class AccessLogMiddleware:
    """Pure ASGI middleware for access logging.
    Logs method, path, status, and duration. Skips noisy healthchecks.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in _SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        start = time.monotonic()
        status_code = 0

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        client = scope.get("client", ("?",))[0] if scope.get("client") else "?"
        method = scope.get("method", "?")
        path = scope["path"]
        logger.info("%s %s %s -> %d (%d ms)", client, method, path, status_code, elapsed_ms)
