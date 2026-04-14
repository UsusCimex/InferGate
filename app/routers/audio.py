from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, JSONResponse

from app.schemas.audio import SpeechRequest
from app.dependencies import get_provider_manager, get_gpu_scheduler, get_cache_manager, get_defaults

router = APIRouter()

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
}


@router.post("/v1/audio/speech")
async def create_speech(
    body: SpeechRequest,
    request: Request,
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    cache=Depends(get_cache_manager),
    defaults=Depends(get_defaults),
):
    model_id = body.model or defaults.get("tts")
    if not model_id:
        return JSONResponse({"error": {"message": "No model specified"}}, status_code=400)

    start = time.monotonic()
    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    params = {
        "voice": body.voice,
        "speed": body.speed,
        "output_format": body.response_format,
    }

    # Cache check
    no_cache = request.headers.get("X-InferGate-No-Cache", "").lower() == "true"
    cache_cfg = config.cache.model_dump()
    should_cache = not no_cache and cache.should_cache(cache_cfg, params)
    cache_key = cache.make_key(model_id, {"input": body.input, **params})
    cache_status = "DISABLED"

    if should_cache:
        cached = await cache.get(cache_key)
        if cached:
            elapsed = int((time.monotonic() - start) * 1000)
            content_type = CONTENT_TYPES.get(body.response_format, "application/octet-stream")
            return Response(
                content=cached,
                media_type=content_type,
                headers={
                    "X-InferGate-Cache": "HIT",
                    "X-InferGate-Model": model_id,
                    "X-InferGate-Generation-Ms": str(elapsed),
                },
            )
        cache_status = "MISS"
        await cache.record_miss(model_id)
    elif no_cache:
        cache_status = "SKIP"

    # Generate
    timeout = config.queue.timeout_seconds
    priority = config.queue.priority

    audio_bytes = await scheduler.submit(
        model_id, priority, provider.synthesize(body.input, **params), timeout
    )

    if should_cache:
        await cache.put(cache_key, audio_bytes, model_id, cache_cfg)

    elapsed = int((time.monotonic() - start) * 1000)
    content_type = CONTENT_TYPES.get(body.response_format, "application/octet-stream")
    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "X-InferGate-Cache": cache_status,
            "X-InferGate-Model": model_id,
            "X-InferGate-Queue-Position": str(scheduler.last_position),
            "X-InferGate-Generation-Ms": str(elapsed),
        },
    )
