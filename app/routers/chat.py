from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.chat import ChatCompletionRequest
from app.dependencies import get_provider_manager, get_gpu_scheduler, get_cache_manager, get_defaults

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    manager = get_provider_manager()
    scheduler = get_gpu_scheduler()
    cache = get_cache_manager()
    defaults = get_defaults()

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

    # Cache check
    no_cache = request.headers.get("X-InferGate-No-Cache", "").lower() == "true"
    cache_cfg = config.cache.model_dump()
    should_cache = not no_cache and cache.should_cache(cache_cfg, params)
    cache_key = cache.make_key(model_id, {"messages": [m.model_dump() for m in body.messages], **params})
    cache_status = "DISABLED"

    if should_cache:
        import json
        cached = await cache.get(cache_key)
        if cached:
            elapsed = int((time.monotonic() - start) * 1000)
            result = json.loads(cached)
            return JSONResponse(
                result,
                headers={
                    "X-InferGate-Cache": "HIT",
                    "X-InferGate-Model": model_id,
                    "X-InferGate-Generation-Ms": str(elapsed),
                },
            )
        cache_status = "MISS"
    elif no_cache:
        cache_status = "SKIP"

    # Run inference through scheduler
    timeout = config.queue.timeout_seconds
    priority = config.queue.priority
    messages = [m.model_dump() for m in body.messages]

    result = await scheduler.submit(
        model_id, priority, provider.generate(messages, **params), timeout
    )

    # Cache result
    if should_cache:
        import json
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
