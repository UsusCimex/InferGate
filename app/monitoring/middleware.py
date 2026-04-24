from __future__ import annotations

import time
import uuid

from starlette.types import ASGIApp, Receive, Scope, Send

from app.monitoring.metrics import (
    _PROMETHEUS_AVAILABLE,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
)


class RequestIdMiddleware:
    """Attach a unique X-Request-ID header and scope.state value to each request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:16]
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = request_id

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append([b"x-request-id", request_id.encode()])
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


_STATIC_ROUTES = frozenset({
    "/v1/chat/completions",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/images/upscale",
    "/v1/audio/speech",
    "/v1/audio/speech/voice-clone",
    "/v1/audio/transcriptions",
    "/v1/models",
    "/cache",
    "/cache/stats",
    "/health",
    "/metrics",
    "/metrics/prometheus",
})


def _normalize_path(path: str) -> str:
    """Collapse dynamic path segments into placeholders so metric labels stay bounded."""
    if path in _STATIC_ROUTES:
        return path
    # Preserve the trailing action so /load vs /unload stay distinct.
    if path.startswith("/v1/models/"):
        parts = path.split("/")
        if len(parts) >= 5:
            return f"/v1/models/{{id}}/{parts[4]}"
        return "/v1/models/{id}"
    if path.startswith("/cache/stats/"):
        return "/cache/stats/{model_id}"
    if path.startswith("/cache/entry/"):
        return "/cache/entry/{key}"
    # Keep /cache/<model_id> distinct from /cache (wipe-all).
    if path.startswith("/cache/"):
        return "/cache/{model_id}"
    return path


class PrometheusMiddleware:
    """Record every HTTP request's count and duration into Prometheus metrics."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _PROMETHEUS_AVAILABLE:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        path = _normalize_path(scope["path"])
        start = time.monotonic()
        status_code = 500

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.monotonic() - start
            REQUESTS_TOTAL.labels(method=method, endpoint=path, status_code=status_code).inc()
            REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
