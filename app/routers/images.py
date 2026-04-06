from __future__ import annotations

import base64
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.images import ImageGenerationRequest, ImageGenerationResponse, ImageData
from app.dependencies import get_provider_manager, get_gpu_scheduler, get_cache_manager, get_defaults

router = APIRouter()


@router.post("/v1/images/generations")
async def generate_images(body: ImageGenerationRequest, request: Request):
    manager = get_provider_manager()
    scheduler = get_gpu_scheduler()
    cache = get_cache_manager()
    defaults = get_defaults()

    model_id = body.model or defaults.get("image")
    if not model_id:
        return JSONResponse({"error": {"message": "No model specified"}}, status_code=400)

    start = time.monotonic()
    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    params: dict = {"size": body.size}
    if body.seed is not None:
        params["seed"] = body.seed

    # Cache check
    no_cache = request.headers.get("X-InferGate-No-Cache", "").lower() == "true"
    cache_cfg = config.cache.model_dump()
    should_cache = not no_cache and cache.should_cache(cache_cfg, params)
    cache_key = cache.make_key(model_id, {"prompt": body.prompt, **params})
    cache_status = "DISABLED"

    if should_cache:
        cached = await cache.get(cache_key)
        if cached:
            elapsed = int((time.monotonic() - start) * 1000)
            b64 = base64.b64encode(cached).decode()
            resp = ImageGenerationResponse(
                created=int(time.time()),
                data=[ImageData(b64_json=b64)],
            )
            return JSONResponse(
                resp.model_dump(),
                headers={
                    "X-InferGate-Cache": "HIT",
                    "X-InferGate-Model": model_id,
                    "X-InferGate-Generation-Ms": str(elapsed),
                },
            )
        cache_status = "MISS"
    elif no_cache:
        cache_status = "SKIP"

    # Generate
    timeout = config.queue.timeout_seconds
    priority = config.queue.priority

    data_list = []
    for _ in range(body.n):
        png_bytes = await scheduler.submit(
            model_id, priority, provider.generate(body.prompt, **params), timeout
        )
        if should_cache:
            await cache.put(cache_key, png_bytes, model_id, cache_cfg)

        if body.response_format == "b64_json":
            data_list.append(ImageData(b64_json=base64.b64encode(png_bytes).decode()))
        else:
            data_list.append(ImageData(b64_json=base64.b64encode(png_bytes).decode()))

    elapsed = int((time.monotonic() - start) * 1000)
    resp = ImageGenerationResponse(created=int(time.time()), data=data_list)
    return JSONResponse(
        resp.model_dump(),
        headers={
            "X-InferGate-Cache": cache_status,
            "X-InferGate-Model": model_id,
            "X-InferGate-Queue-Position": str(scheduler.last_position),
            "X-InferGate-Generation-Ms": str(elapsed),
        },
    )
