from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

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

    global_cache = cache_stats.get("global", {})
    total_hits = 0
    total_misses = 0
    for model_stats in cache_stats.get("per_model", {}).values():
        total_hits += model_stats.get("hit_count", 0)
        total_misses += model_stats.get("miss_count", 0)
    hit_rate = round(total_hits / (total_hits + total_misses) * 100, 1) if (total_hits + total_misses) > 0 else 0.0

    # Try to get GPU info
    gpu_vram_used = 0
    gpu_vram_total = 0
    try:
        import torch

        if torch.cuda.is_available():
            gpu_vram_used = round(torch.cuda.memory_allocated() / (1024 * 1024))
            gpu_vram_total = round(torch.cuda.get_device_properties(0).total_mem / (1024 * 1024))
    except ImportError:
        pass

    return {
        "queue_size": queue_info["queue_size"],
        "max_queue_size": queue_info["max_queue_size"],
        "gpu_vram_used_mb": gpu_vram_used,
        "gpu_vram_total_mb": gpu_vram_total,
        "loaded_models": manager.loaded_models(),
        "cache_hit_rate_percent": hit_rate,
        "uptime_seconds": int(time.time() - start_time),
    }
