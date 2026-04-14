from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from app.schemas.chat import ChatCompletionRequest
from app.dependencies import get_provider_manager, get_gpu_scheduler, get_cache_manager, get_defaults
from app.monitoring import is_prometheus_available, CACHE_HITS, CACHE_MISSES, INFERENCE_DURATION

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    cache=Depends(get_cache_manager),
    defaults=Depends(get_defaults),
):
    model_id = body.model or defaults.get("text")
    if not model_id:
        return JSONResponse({"error": {"message": "No model specified"}}, status_code=400)

    start = time.monotonic()
    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    params = {}
    if body.temperature is not None:
        params["temperature"] = body.temperature
    if body.top_p is not None:
        params["top_p"] = body.top_p
    if body.max_tokens is not None:
        params["max_tokens"] = body.max_tokens
    if body.response_format is not None:
        params["response_format"] = body.response_format.type
    if body.thinking is not None:
        params["thinking"] = body.thinking

    # Streaming mode
    if body.stream:
        messages = [m.model_dump() for m in body.messages]
        return StreamingResponse(
            provider.generate_stream(messages, **params),
            media_type="text/event-stream",
            headers={
                "X-InferGate-Model": model_id,
                "Cache-Control": "no-cache",
            },
        )

    # Cache check
    no_cache = request.headers.get("X-InferGate-No-Cache", "").lower() == "true"
    cache_cfg = config.cache.model_dump()
    should_cache = not no_cache and cache.should_cache(cache_cfg, params)
    cache_key = cache.make_key(model_id, {"messages": [m.model_dump() for m in body.messages], **params})
    cache_status = "DISABLED"

    if should_cache:
        cached = await cache.get(cache_key)
        if cached:
            elapsed = int((time.monotonic() - start) * 1000)
            try:
                result = json.loads(cached)
            except (json.JSONDecodeError, ValueError):
                await cache.invalidate_key(cache_key)
            else:
                if is_prometheus_available():
                    CACHE_HITS.labels(model_id=model_id).inc()
                return JSONResponse(
                    result,
                    headers={
                        "X-InferGate-Cache": "HIT",
                        "X-InferGate-Model": model_id,
                        "X-InferGate-Generation-Ms": str(elapsed),
                    },
                )
        cache_status = "MISS"
        await cache.record_miss(model_id)
        if is_prometheus_available():
            CACHE_MISSES.labels(model_id=model_id).inc()
    elif no_cache:
        cache_status = "SKIP"

    # Run inference through scheduler
    timeout = config.queue.timeout_seconds
    priority = config.queue.priority
    messages = [m.model_dump() for m in body.messages]

    inference_start = time.monotonic()
    result = await scheduler.submit(
        model_id, priority, provider.generate(messages, **params), timeout
    )
    if is_prometheus_available():
        INFERENCE_DURATION.labels(model_id=model_id, category="text").observe(
            time.monotonic() - inference_start
        )

    # Cache result
    if should_cache:
        await cache.put(cache_key, json.dumps(result).encode(), model_id, cache_cfg)

    elapsed = int((time.monotonic() - start) * 1000)
    return JSONResponse(
        result,
        headers={
            "X-InferGate-Cache": cache_status,
            "X-InferGate-Model": model_id,
            "X-InferGate-Queue-Position": str(scheduler.last_position),
            "X-InferGate-Generation-Ms": str(elapsed),
        },
    )
