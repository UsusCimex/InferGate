"""Standalone FastAPI worker that loads and serves a single model.

Usage:
    WORKER_MODEL_CONFIG=config/models/qwen3.5-4b.yaml \
    uvicorn app.worker:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import logging
import os
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.responses import StreamingResponse

from app.config import load_single_model_config
from app.providers.base import BaseProvider
from app.providers.registry import get_provider_class

logger = logging.getLogger("infergate.worker")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("WORKER_MODEL_CONFIG", "config/models/model.yaml")
    models_dir = os.environ.get("WORKER_MODELS_DIR", "./models")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    config = load_single_model_config(config_path)
    logger.info("Worker starting for model: %s (%s)", config.id, config.provider_class)

    # GPU compatibility check
    vram_mb = config.model.get("vram_mb", 0)
    if vram_mb > 0:
        try:
            import torch
            if not torch.cuda.is_available():
                logger.error("CUDA not available — cannot load GPU model %s", config.id)
                raise RuntimeError("CUDA is not available")
            gpu_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            logger.info(
                "GPU: %s (compute capability %d.%d), CUDA %s, PyTorch %s",
                gpu_name, cc[0], cc[1], torch.version.cuda, torch.__version__,
            )
        except ImportError:
            logger.warning("torch not available — skipping GPU check")

    provider_cls = get_provider_class(config.provider_class)
    provider = provider_cls(config)
    await provider.load(models_dir)

    app.state.provider = provider
    app.state.config = config

    logger.info("Worker ready: %s", config.id)
    yield

    logger.info("Worker shutting down: %s", config.id)
    await provider.unload()


app = FastAPI(title="InferGate Worker", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    provider: BaseProvider = request.app.state.provider
    config = request.app.state.config
    return {
        "status": "ok" if provider.is_loaded() else "loading",
        "model": config.id,
        "category": config.category,
    }


@app.post("/load")
async def load(request: Request):
    """Explicit load signal from gateway. Reloads model if it was previously unloaded."""
    provider: BaseProvider = request.app.state.provider
    if not provider.is_loaded():
        models_dir = os.environ.get("WORKER_MODELS_DIR", "./models")
        await provider.load(models_dir)
        logger.info("Reloaded %s via /load", request.app.state.config.id)
    return {"status": "ok", "model": request.app.state.config.id}


@app.post("/unload")
async def unload(request: Request):
    provider: BaseProvider = request.app.state.provider
    await provider.unload()
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: Request):
    """Generate text or image depending on model category."""
    provider: BaseProvider = request.app.state.provider
    config = request.app.state.config
    body = await request.json()

    if config.category == "text":
        messages = body.pop("messages")
        stream = body.pop("stream", False)

        if stream and hasattr(provider, "generate_stream"):
            return StreamingResponse(
                provider.generate_stream(messages, **body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        result = await provider.generate(messages, **body)
        return JSONResponse(result)

    elif config.category == "image":
        prompt = body.pop("prompt")
        png_bytes = await provider.generate(prompt, **body)
        return Response(content=png_bytes, media_type="image/png")

    return JSONResponse(
        {"error": {"message": f"Unknown category: {config.category}"}},
        status_code=400,
    )


@app.post("/synthesize")
async def synthesize(request: Request):
    """Synthesize speech."""
    provider: BaseProvider = request.app.state.provider
    body = await request.json()

    text = body.pop("text")
    audio_bytes = await provider.synthesize(text, **body)
    return Response(content=audio_bytes, media_type="application/octet-stream")
