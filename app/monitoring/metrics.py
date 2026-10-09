from __future__ import annotations

import contextlib

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

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
    WORKER_UP = Gauge(
        "infergate_worker_up",
        "1 when the last /health probe of the model's worker answered 200",
        labelnames=["model_id"],
    )
    WORKER_HEALTH_CHECK_DURATION = Histogram(
        "infergate_worker_health_check_duration_seconds",
        "Duration of the worker /health probes in seconds",
        labelnames=["model_id"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 3],
    )
    WORKER_DISCONNECTS = Counter(
        "infergate_worker_disconnects_total",
        "Loaded models the worker monitor dropped",
        labelnames=["model_id", "reason"],
    )
    HTTP_POOL_CONNECTIONS = Gauge(
        "infergate_http_pool_connections",
        "Open connections of the gateway's HTTP pool to a worker",
        labelnames=["model_id", "state"],
    )
    HTTP_POOL_WAITING = Gauge(
        "infergate_http_pool_waiting_requests",
        "Requests waiting for a free connection to a worker",
        labelnames=["model_id"],
    )
    EMBEDDING_BATCH_INPUTS = Histogram(
        "infergate_embedding_batch_inputs",
        "Inputs per micro-batched /v1/embeddings call",
        labelnames=["model_id"],
        buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
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
    WORKER_UP = None  # type: ignore[assignment]
    WORKER_HEALTH_CHECK_DURATION = None  # type: ignore[assignment]
    WORKER_DISCONNECTS = None  # type: ignore[assignment]
    HTTP_POOL_CONNECTIONS = None  # type: ignore[assignment]
    HTTP_POOL_WAITING = None  # type: ignore[assignment]
    EMBEDDING_BATCH_INPUTS = None  # type: ignore[assignment]


def is_prometheus_available() -> bool:
    return _PROMETHEUS_AVAILABLE


def record_worker_probe(model_id: str, healthy: bool, seconds: float) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    WORKER_UP.labels(model_id=model_id).set(1 if healthy else 0)
    WORKER_HEALTH_CHECK_DURATION.labels(model_id=model_id).observe(seconds)


def record_worker_disconnect(model_id: str, reason: str) -> None:
    if _PROMETHEUS_AVAILABLE:
        WORKER_DISCONNECTS.labels(model_id=model_id, reason=reason).inc()


def forget_worker(model_id: str) -> None:
    """Drop the up gauge of a model the worker monitor no longer watches."""
    if not _PROMETHEUS_AVAILABLE:
        return
    with contextlib.suppress(KeyError):
        WORKER_UP.remove(model_id)


def update_pool_gauges(pools: dict[str, dict[str, int]]) -> None:
    """Publish per-worker HTTP pool counts: model id to `active`, `idle` and `waiting`."""
    if not _PROMETHEUS_AVAILABLE:
        return
    HTTP_POOL_CONNECTIONS.clear()
    HTTP_POOL_WAITING.clear()
    for model_id, stats in pools.items():
        HTTP_POOL_CONNECTIONS.labels(model_id=model_id, state="active").set(stats["active"])
        HTTP_POOL_CONNECTIONS.labels(model_id=model_id, state="idle").set(stats["idle"])
        HTTP_POOL_WAITING.labels(model_id=model_id).set(stats["waiting"])


def record_embedding_batch(model_id: str, inputs: int) -> None:
    if _PROMETHEUS_AVAILABLE:
        EMBEDDING_BATCH_INPUTS.labels(model_id=model_id).observe(inputs)


def update_runtime_gauges(
    models_loaded: int,
    gpu_vram_used_mb: int,
    queue_size: int,
) -> None:
    """Publish runtime gauges (no-op when prometheus-client is absent)."""
    if not _PROMETHEUS_AVAILABLE:
        return
    MODELS_LOADED.set(models_loaded)
    GPU_VRAM_USED_MB.set(gpu_vram_used_mb)
    QUEUE_SIZE.set(queue_size)
