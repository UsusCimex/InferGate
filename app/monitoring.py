"""Prometheus metrics and request ID tracking for observability."""
from __future__ import annotations

import time
import uuid

from starlette.types import ASGIApp, Receive, Scope, Send

# --- Request ID middleware ---


class RequestIdMiddleware:
    """Pure ASGI middleware that assigns a unique request ID to each request.
    Stored in scope["state"]["request_id"] and returned as X-Request-ID header.
    """

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


# --- Prometheus metrics ---

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

    REQUESTS_TOTAL = Counter(
        "infergate_requests_total",
        "Total HTTP requests",
        labelnames=["method", "endpoint", "status_code"],
    )
    REQUEST_DURATION = Histogram(
        "infergate_request_duration_seconds",
        "Request duration in seconds",
        labelnames=["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120],
    )
    MODELS_LOADED = Gauge(
        "infergate_models_loaded",
        "Number of currently loaded models",
    )
    GPU_VRAM_USED_MB = Gauge(
        "infergate_gpu_vram_used_mb",
        "GPU VRAM used in megabytes",
    )
    QUEUE_SIZE = Gauge(
        "infergate_queue_size",
        "Current GPU scheduler queue size",
    )
    CACHE_HITS = Counter(
        "infergate_cache_hits_total",
        "Cache hits",
        labelnames=["model_id"],
    )
    CACHE_MISSES = Counter(
        "infergate_cache_misses_total",
        "Cache misses",
        labelnames=["model_id"],
    )
    INFERENCE_DURATION = Histogram(
        "infergate_inference_duration_seconds",
        "Model inference duration in seconds",
        labelnames=["model_id", "category"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
    )

    _PROMETHEUS_AVAILABLE = True

except ImportError:
    _PROMETHEUS_AVAILABLE = False
    generate_latest = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain"
    REQUESTS_TOTAL = None  # type: ignore[assignment]
    REQUEST_DURATION = None  # type: ignore[assignment]
    MODELS_LOADED = None  # type: ignore[assignment]
    GPU_VRAM_USED_MB = None  # type: ignore[assignment]
    QUEUE_SIZE = None  # type: ignore[assignment]
    CACHE_HITS = None  # type: ignore[assignment]
    CACHE_MISSES = None  # type: ignore[assignment]
    INFERENCE_DURATION = None  # type: ignore[assignment]


def is_prometheus_available() -> bool:
    return _PROMETHEUS_AVAILABLE


# Route normalization for metric labels
_ROUTE_MAP = {
    "/v1/chat/completions": "/v1/chat/completions",
    "/v1/images/generations": "/v1/images/generations",
    "/v1/audio/speech": "/v1/audio/speech",
    "/v1/models": "/v1/models",
    "/health": "/health",
    "/metrics": "/metrics",
}


def _normalize_path(path: str) -> str:
    """Normalize path to avoid high cardinality in metrics."""
    if path in _ROUTE_MAP:
        return path
    if path.startswith("/v1/models/"):
        return "/v1/models/{id}"
    if path.startswith("/cache/stats/"):
        return "/cache/stats/{model_id}"
    if path.startswith("/cache/entry/"):
        return "/cache/entry/{key}"
    if path.startswith("/cache"):
        return "/cache"
    return path


class PrometheusMiddleware:
    """Pure ASGI middleware that records request count and duration."""

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
