from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.dependencies import get_defaults, get_gpu_scheduler, get_provider_manager
from app.monitoring import INFERENCE_DURATION, is_prometheus_available
from app.schemas.embeddings import (
    AudioEmbeddingResponse,
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageEmbeddingResponse,
    VideoEmbeddingResponse,
)

router = APIRouter()

_MAX_AUDIO_BYTES = 100 * 1024 * 1024
_MAX_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_VIDEO_BYTES = 200 * 1024 * 1024


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    body: EmbeddingRequest,
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    defaults=Depends(get_defaults),
):
    """OpenAI-compatible text-embeddings endpoint."""
    model_id = body.model or defaults.get("embedding_text")
    if not model_id:
        return JSONResponse(
            {"error": {"message": "No embedding model specified"}}, status_code=400
        )

    inputs = [body.input] if isinstance(body.input, str) else body.input
    if not inputs:
        return JSONResponse(
            {"error": {"message": "input must be non-empty"}}, status_code=400
        )

    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    inference_start = time.monotonic()
    async with manager.active_request(model_id):
        vecs = await scheduler.submit(
            model_id,
            config.queue.priority,
            provider.embed(inputs),
            config.queue.timeout_seconds,
        )
    if is_prometheus_available():
        INFERENCE_DURATION.labels(model_id=model_id, category="embedding-text").observe(
            time.monotonic() - inference_start
        )

    return EmbeddingResponse(
        model=model_id,
        data=[EmbeddingItem(embedding=v, index=i) for i, v in enumerate(vecs)],
        usage={"prompt_tokens": 0, "total_tokens": 0},
    )


@router.post("/v1/embeddings/audio", response_model=AudioEmbeddingResponse)
async def create_audio_embedding(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    defaults=Depends(get_defaults),
):
    """Encode an audio file (≤100MB) into a single embedding vector."""
    model_id = model or defaults.get("embedding_audio")
    if not model_id:
        return JSONResponse(
            {"error": {"message": "No audio embedding model specified"}}, status_code=400
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        return JSONResponse({"error": {"message": "Empty audio file"}}, status_code=400)
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        return JSONResponse(
            {"error": {"message": f"Audio exceeds {_MAX_AUDIO_BYTES // (1024 * 1024)}MB limit"}},
            status_code=413,
        )

    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    inference_start = time.monotonic()
    async with manager.active_request(model_id):
        vec = await scheduler.submit(
            model_id,
            config.queue.priority,
            provider.embed(audio_bytes, filename=file.filename or "audio.wav"),
            config.queue.timeout_seconds,
        )
    if is_prometheus_available():
        INFERENCE_DURATION.labels(model_id=model_id, category="embedding-audio").observe(
            time.monotonic() - inference_start
        )

    return AudioEmbeddingResponse(model=model_id, embedding=vec)


@router.post("/v1/embeddings/image", response_model=ImageEmbeddingResponse)
async def create_image_embedding(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    defaults=Depends(get_defaults),
):
    """Encode an image (≤25MB) into a single embedding vector."""
    model_id = model or defaults.get("embedding_image")
    if not model_id:
        return JSONResponse(
            {"error": {"message": "No image embedding model specified"}}, status_code=400
        )

    image_bytes = await file.read()
    if not image_bytes:
        return JSONResponse({"error": {"message": "Empty image file"}}, status_code=400)
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        return JSONResponse(
            {"error": {"message": f"Image exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)}MB limit"}},
            status_code=413,
        )

    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    inference_start = time.monotonic()
    async with manager.active_request(model_id):
        vec = await scheduler.submit(
            model_id,
            config.queue.priority,
            provider.embed_image(image_bytes, filename=file.filename or "image.jpg"),
            config.queue.timeout_seconds,
        )
    if is_prometheus_available():
        INFERENCE_DURATION.labels(model_id=model_id, category="embedding-image").observe(
            time.monotonic() - inference_start
        )

    return ImageEmbeddingResponse(model=model_id, embedding=vec)


@router.post("/v1/embeddings/video", response_model=VideoEmbeddingResponse)
async def create_video_embedding(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    defaults=Depends(get_defaults),
):
    """Encode a video clip (≤200MB) into a single embedding vector."""
    model_id = model or defaults.get("embedding_video")
    if not model_id:
        return JSONResponse(
            {"error": {"message": "No video embedding model specified"}}, status_code=400
        )

    video_bytes = await file.read()
    if not video_bytes:
        return JSONResponse({"error": {"message": "Empty video file"}}, status_code=400)
    if len(video_bytes) > _MAX_VIDEO_BYTES:
        return JSONResponse(
            {"error": {"message": f"Video exceeds {_MAX_VIDEO_BYTES // (1024 * 1024)}MB limit"}},
            status_code=413,
        )

    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    inference_start = time.monotonic()
    async with manager.active_request(model_id):
        vec = await scheduler.submit(
            model_id,
            config.queue.priority,
            provider.embed_video(video_bytes, filename=file.filename or "clip.mp4"),
            config.queue.timeout_seconds,
        )
    if is_prometheus_available():
        INFERENCE_DURATION.labels(model_id=model_id, category="embedding-video").observe(
            time.monotonic() - inference_start
        )

    return VideoEmbeddingResponse(model=model_id, embedding=vec)
