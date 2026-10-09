from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from app.dependencies import (
    get_cache_manager,
    get_gpu_scheduler,
    get_provider_manager,
    get_start_time,
)
from app.monitoring import (
    CONTENT_TYPE_LATEST,
    generate_latest,
    is_prometheus_available,
    update_runtime_gauges,
)
from app.services.memory_watchdog import live_vram

router = APIRouter()


@router.get("/health")
@router.get("/v1/health")
async def health(cache=Depends(get_cache_manager)):
    """Liveness probe. /v1/health alias matches OpenAI-style clients that prefix every call."""
    if not cache.is_initialized():
        return JSONResponse({"status": "unhealthy", "reason": "cache DB not initialized"}, status_code=503)
    return {"status": "ok"}


@router.get("/metrics")
async def metrics(
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    cache=Depends(get_cache_manager),
    start_time=Depends(get_start_time),
):
    queue_info = scheduler.queue_info()
    cache_stats = await cache.stats()

    total_hits = 0
    total_misses = 0
    for model_stats in cache_stats.get("per_model", {}).values():
        total_hits += model_stats.get("hit_count", 0)
        total_misses += model_stats.get("miss_count", 0)
    hit_rate = round(total_hits / (total_hits + total_misses) * 100, 1) if (total_hits + total_misses) > 0 else 0.0

    gpu_vram_used, gpu_vram_total = await _publish_gauges(manager, scheduler)

    return {
        "queue_size": queue_info["queue_size"],
        "max_queue_size": queue_info["max_queue_size"],
        "gpu_vram_used_mb": gpu_vram_used,
        "gpu_vram_total_mb": gpu_vram_total,
        "loaded_models": manager.loaded_models(),
        "cache_hit_rate_percent": hit_rate,
        "uptime_seconds": int(time.time() - start_time),
    }


@router.get("/metrics/prometheus")
async def prometheus_metrics(
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
):
    """Return metrics in Prometheus text-exposition format."""
    if not is_prometheus_available():
        return JSONResponse(
            {"error": {"message": "prometheus-client not installed. pip install prometheus-client"}},
            status_code=501,
        )
    await _publish_gauges(manager, scheduler)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _publish_gauges(manager, scheduler) -> tuple[int, int]:
    """Refresh the runtime gauges; returns the used and total GPU MB."""
    used, total = await live_vram(manager)
    if total == 0:
        used, total = _in_process_vram()
    update_runtime_gauges(
        models_loaded=len(manager.loaded_models()),
        gpu_vram_used_mb=used,
        queue_size=scheduler.queue_info()["queue_size"],
    )
    return used, total


def _in_process_vram() -> tuple[int, int]:
    """VRAM of models loaded inside the gateway process; (0, 0) without torch or CUDA."""
    try:
        import torch
    except ImportError:
        return 0, 0
    if not torch.cuda.is_available():
        return 0, 0
    used = round(torch.cuda.memory_allocated() / (1024 * 1024))
    total = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    return used, total
