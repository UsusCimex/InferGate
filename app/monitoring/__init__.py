"""Observability: Prometheus metrics, request ID tracking, middleware."""
from app.monitoring.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    CONTENT_TYPE_LATEST,
    GPU_VRAM_USED_MB,
    INFERENCE_DURATION,
    MODELS_LOADED,
    QUEUE_SIZE,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    generate_latest,
    is_prometheus_available,
    update_runtime_gauges,
)
from app.monitoring.middleware import (
    PrometheusMiddleware,
    RequestIdMiddleware,
    _normalize_path,
)
from app.monitoring.request_context import get_request_id, set_request_id

__all__ = [
    "CACHE_HITS",
    "CACHE_MISSES",
    "CONTENT_TYPE_LATEST",
    "GPU_VRAM_USED_MB",
    "INFERENCE_DURATION",
    "MODELS_LOADED",
    "QUEUE_SIZE",
    "REQUESTS_TOTAL",
    "REQUEST_DURATION",
    "PrometheusMiddleware",
    "RequestIdMiddleware",
    "_normalize_path",
    "generate_latest",
    "get_request_id",
    "is_prometheus_available",
    "set_request_id",
    "update_runtime_gauges",
]
