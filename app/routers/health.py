from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from app.dependencies import get_provider_manager, get_gpu_scheduler, get_cache_manager, get_start_time

router = APIRouter()


@router.get("/health")
async def health(cache=Depends(get_cache_manager)):
    if cache._db is None:
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

    gpu_vram_used = 0
    gpu_vram_total = 0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_vram_used = round(torch.cuda.memory_allocated() / (1024 * 1024))
            gpu_vram_total = round(torch.cuda.get_device_properties(0).total_mem / (1024 * 1024))
    except ImportError:
        pass

    # Update Prometheus gauges if available
    from app.monitoring import is_prometheus_available, MODELS_LOADED, GPU_VRAM_USED_MB, QUEUE_SIZE
    if is_prometheus_available():
        MODELS_LOADED.set(len(manager.loaded_models()))
        GPU_VRAM_USED_MB.set(gpu_vram_used)
        QUEUE_SIZE.set(queue_info["queue_size"])

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
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint for scraping."""
    from app.monitoring import is_prometheus_available, generate_latest, CONTENT_TYPE_LATEST

    if not is_prometheus_available():
        return JSONResponse(
            {"error": {"message": "prometheus-client not installed. pip install prometheus-client"}},
            status_code=501,
        )
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
