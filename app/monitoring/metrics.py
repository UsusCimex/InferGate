"""Prometheus metric definitions. Gracefully degrades if prometheus-client not installed."""
from __future__ import annotations

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
