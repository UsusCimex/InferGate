from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_cache_manager

router = APIRouter()


@router.get("/cache/stats")
async def cache_stats(cache=Depends(get_cache_manager)):
    stats = await cache.stats()
    return stats


@router.get("/cache/stats/{model_id}")
async def cache_stats_model(model_id: str, cache=Depends(get_cache_manager)):
    stats = await cache.stats(model_id)
    return stats


@router.delete("/cache")
async def clear_all_cache(cache=Depends(get_cache_manager)):
    count = await cache.invalidate_all()
    return {"deleted": count, "message": f"Cleared all cache ({count} entries)"}


@router.delete("/cache/{model_id}")
async def clear_model_cache(model_id: str, cache=Depends(get_cache_manager)):
    count = await cache.invalidate_model(model_id)
    return {"deleted": count, "message": f"Cleared cache for {model_id} ({count} entries)"}


@router.delete("/cache/entry/{cache_key}")
async def delete_cache_entry(cache_key: str, cache=Depends(get_cache_manager)):
    found = await cache.invalidate_key(cache_key)
    if not found:
        return JSONResponse(
            {"error": {"message": f"Cache entry '{cache_key}' not found"}},
            status_code=404,
        )
    return {"deleted": 1, "message": "Entry deleted"}
