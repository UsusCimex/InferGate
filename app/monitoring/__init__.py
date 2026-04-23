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
    "is_prometheus_available",
    "update_runtime_gauges",
]
