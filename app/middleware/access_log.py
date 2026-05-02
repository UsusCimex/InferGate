from __future__ import annotations

import json
import logging
import os
import time

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

_SKIP_PATHS = frozenset({"/health", "/openapi.json"})


def _format_enabled() -> bool:
    """Read INFERGATE_ACCESS_LOG_JSON each call so tests can flip it via monkeypatch."""
    return os.environ.get("INFERGATE_ACCESS_LOG_JSON", "").lower() in {"1", "true", "yes"}


class AccessLogMiddleware:
    """Log method, path, status, latency and request id for every HTTP request.

    Default format is human-readable. Set INFERGATE_ACCESS_LOG_JSON=true for
    structured single-line JSON suitable for ingest into Loki / Elastic / etc.
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
        request_id = (scope.get("state") or {}).get("request_id")

        if _format_enabled():
            payload = {
                "msg": "http_access",
                "client": client,
                "method": method,
                "path": path,
                "status": status_code,
                "latency_ms": elapsed_ms,
            }
            if request_id:
                payload["request_id"] = request_id
            logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            return

        if request_id:
            logger.info(
                "%s %s %s -> %d (%d ms) [%s]",
                client, method, path, status_code, elapsed_ms, request_id,
            )
        else:
            logger.info(
                "%s %s %s -> %d (%d ms)",
                client, method, path, status_code, elapsed_ms,
            )
