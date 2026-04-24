from __future__ import annotations

import base64
import hashlib
import time

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.dependencies import (
    get_cache_manager,
    get_defaults,
    get_gpu_scheduler,
    get_provider_manager,
)
from app.monitoring import CACHE_HITS, CACHE_MISSES, INFERENCE_DURATION, is_prometheus_available
from app.schemas.images import ImageData, ImageGenerationRequest, ImageGenerationResponse

router = APIRouter()

_MAX_UPSCALE_BYTES = 50 * 1024 * 1024


@router.post("/v1/images/generations")
async def generate_images(
    body: ImageGenerationRequest,
    request: Request,
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    cache=Depends(get_cache_manager),
    defaults=Depends(get_defaults),
):
    model_id = body.model or defaults.get("image")
    if not model_id:
        return JSONResponse({"error": {"message": "No model specified"}}, status_code=400)

    start = time.monotonic()
    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    params: dict = {"size": body.size}
    if body.seed is not None:
        params["seed"] = body.seed
    if body.negative_prompt is not None:
        params["negative_prompt"] = body.negative_prompt
    if body.num_inference_steps is not None:
        params["num_inference_steps"] = body.num_inference_steps
    if body.guidance_scale is not None:
        params["guidance_scale"] = body.guidance_scale
    if body.scheduler is not None:
        params["scheduler"] = body.scheduler
    if body.loras is not None:
        # Serialise to dicts — LoraSpec models don't auto-serialise across the httpx hop.
        params["loras"] = [lora.model_dump() for lora in body.loras]
    if body.textual_inversions is not None:
        params["textual_inversions"] = [ti.model_dump() for ti in body.textual_inversions]
    if body.highres_fix is not None:
        params["highres_fix"] = body.highres_fix.model_dump()
    if body.image is not None:
        params["image"] = body.image
    if body.mask is not None:
        params["mask"] = body.mask
    if body.denoising_strength is not None:
        params["denoising_strength"] = body.denoising_strength
    if body.refiner_switch_at is not None:
        params["refiner_switch_at"] = body.refiner_switch_at

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
            if is_prometheus_available():
                CACHE_HITS.labels(model_id=model_id).inc()
            return JSONResponse(
                resp.model_dump(),
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

    timeout = config.queue.timeout_seconds
    priority = config.queue.priority

    data_list = []
    for _ in range(body.n):
        inference_start = time.monotonic()
        async with manager.active_request(model_id):
            png_bytes = await scheduler.submit(
                model_id, priority, provider.generate(body.prompt, **params), timeout
            )
        if is_prometheus_available():
            INFERENCE_DURATION.labels(model_id=model_id, category="image").observe(
                time.monotonic() - inference_start
            )
        if should_cache:
            await cache.put(cache_key, png_bytes, model_id, cache_cfg)

        b64 = base64.b64encode(png_bytes).decode()
        if body.response_format == "url":
            data_list.append(ImageData(url=f"data:image/png;base64,{b64}"))
        else:
            data_list.append(ImageData(b64_json=b64))

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


@router.post("/v1/images/edits")
async def edit_images(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=1, max_length=10000),
    mask: UploadFile | None = File(None),
    model: str | None = Form(None),
    n: int = Form(1, ge=1, le=10),
    size: str = Form("1024x1024"),
    response_format: str = Form("b64_json"),
    seed: int | None = Form(None),
    negative_prompt: str | None = Form(None),
    num_inference_steps: int | None = Form(None, ge=1, le=150),
    guidance_scale: float | None = Form(None, ge=0.0, le=30.0),
    scheduler_name: str | None = Form(None, alias="scheduler"),
    denoising_strength: float | None = Form(None, ge=0.0, le=1.0),
    manager=Depends(get_provider_manager),
    scheduler_dep=Depends(get_gpu_scheduler),
    cache=Depends(get_cache_manager),
    defaults=Depends(get_defaults),
):
    """Multipart img2img/inpaint — delegates to the same path as /v1/images/generations."""
    image_bytes = await image.read()
    if not image_bytes:
        return JSONResponse({"error": {"message": "Empty image file"}}, status_code=400)

    mask_bytes: bytes | None = None
    if mask is not None and mask.filename:
        mask_bytes = await mask.read()
        if not mask_bytes:
            mask_bytes = None

    try:
        body = ImageGenerationRequest(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            scheduler=scheduler_name,
            denoising_strength=denoising_strength,
            image=base64.b64encode(image_bytes).decode(),
            mask=base64.b64encode(mask_bytes).decode() if mask_bytes else None,
        )
    except ValueError as e:
        return JSONResponse(
            {"error": {"message": f"invalid request: {e}", "type": "invalid_request"}},
            status_code=422,
        )

    return await generate_images(
        body=body, request=request,
        manager=manager, scheduler=scheduler_dep, cache=cache, defaults=defaults,
    )


@router.post("/v1/images/upscale")
async def upscale_image(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(None),
    response_format: str = Form("b64_json"),
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    cache=Depends(get_cache_manager),
    defaults=Depends(get_defaults),
):
    """Upscale an uploaded image; returns b64_json (default) or raw PNG bytes."""
    if response_format not in {"b64_json", "png"}:
        return JSONResponse(
            {"error": {"message": "response_format must be 'b64_json' or 'png'"}},
            status_code=400,
        )

    model_id = model or defaults.get("upscale")
    if not model_id:
        return JSONResponse({"error": {"message": "No upscale model specified"}}, status_code=400)

    image_bytes = await file.read()
    if not image_bytes:
        return JSONResponse({"error": {"message": "Empty image file"}}, status_code=400)
    if len(image_bytes) > _MAX_UPSCALE_BYTES:
        return JSONResponse(
            {"error": {"message": f"Image exceeds {_MAX_UPSCALE_BYTES // (1024*1024)}MB limit"}},
            status_code=413,
        )

    start = time.monotonic()
    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    no_cache = request.headers.get("X-InferGate-No-Cache", "").lower() == "true"
    cache_cfg = config.cache.model_dump()
    sha = hashlib.sha256(image_bytes).hexdigest()[:32]
    params = {"sha": sha, "size": len(image_bytes)}
    should_cache = not no_cache and cache.should_cache(cache_cfg, params)
    cache_key = cache.make_key(model_id, params)
    cache_status = "DISABLED"

    if should_cache:
        cached = await cache.get(cache_key)
        if cached:
            elapsed = int((time.monotonic() - start) * 1000)
            if is_prometheus_available():
                CACHE_HITS.labels(model_id=model_id).inc()
            return _upscale_response(cached, response_format, model_id, elapsed, "HIT",
                                      scheduler.last_position)
        cache_status = "MISS"
        await cache.record_miss(model_id)
        if is_prometheus_available():
            CACHE_MISSES.labels(model_id=model_id).inc()
    elif no_cache:
        cache_status = "SKIP"

    timeout = config.queue.timeout_seconds
    priority = config.queue.priority

    inference_start = time.monotonic()
    async with manager.active_request(model_id):
        png_bytes = await scheduler.submit(
            model_id, priority, provider.upscale(image_bytes), timeout
        )
    if is_prometheus_available():
        INFERENCE_DURATION.labels(model_id=model_id, category="upscale").observe(
            time.monotonic() - inference_start
        )

    if should_cache:
        await cache.put(cache_key, png_bytes, model_id, cache_cfg)

    elapsed = int((time.monotonic() - start) * 1000)
    return _upscale_response(png_bytes, response_format, model_id, elapsed, cache_status,
                              scheduler.last_position)


def _upscale_response(
    png_bytes: bytes, response_format: str, model_id: str, elapsed_ms: int,
    cache_status: str, queue_position: int,
):
    headers = {
        "X-InferGate-Cache": cache_status,
        "X-InferGate-Model": model_id,
        "X-InferGate-Queue-Position": str(queue_position),
        "X-InferGate-Generation-Ms": str(elapsed_ms),
    }
    if response_format == "png":
        return Response(content=png_bytes, media_type="image/png", headers=headers)
    b64 = base64.b64encode(png_bytes).decode()
    return JSONResponse(
        {"created": int(time.time()), "data": [{"b64_json": b64}]},
        headers=headers,
    )
