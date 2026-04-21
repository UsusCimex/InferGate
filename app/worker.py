"""Standalone FastAPI worker that loads and serves a single model.

Usage:
    WORKER_MODEL_CONFIG=config/models/qwen3.5-4b.yaml \
    uvicorn app.worker:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import asyncio
import logging
import os
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.responses import StreamingResponse

from app.config import ModelConfig, load_single_model_config
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
    # Serialises /reload against /generate and /synthesize. For most
    # models this is a non-issue because their YAML has max_concurrent=1,
    # so requests are already sequential. For models with
    # max_concurrent>1 (e.g. kokoro TTS), a reload briefly serialises
    # inflight work until the swap completes — acceptable tradeoff
    # for a rare operator-driven operation vs. the complexity of a
    # refcount-based drain protocol. See POST /reload docstring.
    app.state.reload_lock = asyncio.Lock()

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


@app.post("/reload")
async def reload_config(request: Request):
    """Hot-swap the worker's model config without restarting the container.

    Gateway calls this from `ProviderManager.reload_model()` when it
    detects a YAML change on disk and the change touches the
    worker-observable part of the config (`model.*`, `provider_class`,
    `worker_url`). Pure metadata changes (display_name, description,
    cache strategy) are handled gateway-side and never hit the worker.

    Classification of changes:
      * identical config              → no-op (action="noop")
      * only metadata differs         → update state.config in place
                                        so subsequent /generate sees
                                        new defaults (action="metadata")
      * `model.*` / provider_class /   → unload + build fresh provider
         worker_url differs             + load, swap app.state.provider
                                        atomically (action="full_reload")

    Concurrency: takes `reload_lock` for the whole operation, which is
    also held by `/generate` and `/synthesize`. Inflight requests
    finish before the reload starts, and new requests wait until the
    swap completes. For single-concurrent models (most diffusers
    pipelines) this is a no-op; for concurrent TTS / text models the
    reload briefly serialises requests — acceptable because reload is
    operator-driven and rare.

    Errors:
      * 400 on malformed config or unknown provider_class
      * 500 if the new provider's `load()` raises — in that case the
        old provider is kept active so the worker stays serviceable.
    """
    body = await request.json()
    try:
        new_config = ModelConfig(**body)
    except Exception as e:  # noqa: BLE001 — pydantic + value errors both → 400
        return JSONResponse(
            {"error": {"message": f"invalid config: {e}", "type": "invalid_request"}},
            status_code=400,
        )

    async with request.app.state.reload_lock:
        old_provider: BaseProvider = request.app.state.provider
        old_config: ModelConfig = request.app.state.config

        if old_config.model_dump() == new_config.model_dump():
            return {"status": "ok", "action": "noop", "model": new_config.id}

        # Decide between metadata-only and full reload. The worker only
        # cares about things that affect inference: the model section
        # (weights, defaults, quantization, device) and provider_class.
        # Everything else — display_name, description, cache, queue
        # priority/timeout — is gateway-side and doesn't need a swap here.
        full_reload_needed = (
            old_config.provider_class != new_config.provider_class
            or old_config.model != new_config.model
            or old_config.worker_url != new_config.worker_url
        )

        if not full_reload_needed:
            request.app.state.config = new_config
            # Point the existing provider at the new config so the next
            # /generate sees the updated default_params, cache strategy, etc.
            old_provider.config = new_config
            logger.info("Reloaded %s (metadata only)", new_config.id)
            return {"status": "ok", "action": "metadata", "model": new_config.id}

        # Full reload: load new before touching old so the worker stays
        # serviceable if the new config is bad (e.g. bogus hub_id).
        models_dir = os.environ.get("WORKER_MODELS_DIR", "./models")
        try:
            provider_cls = get_provider_class(new_config.provider_class)
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request"}},
                status_code=400,
            )

        new_provider = provider_cls(new_config)
        logger.info("Reloading %s (full): loading new provider …", new_config.id)
        try:
            await new_provider.load(models_dir)
        except Exception as e:  # noqa: BLE001 — surface as 500, keep worker alive
            logger.error("Failed to load new provider during reload: %s", e)
            return JSONResponse(
                {"error": {"message": f"load failed: {e}", "type": "load_failed"}},
                status_code=500,
            )

        # Swap + unload old. Everything waits on reload_lock, so no
        # inflight /generate is using old_provider at this point.
        request.app.state.provider = new_provider
        request.app.state.config = new_config
        try:
            await old_provider.unload()
        except Exception as e:  # noqa: BLE001 — best-effort cleanup
            logger.warning("Error unloading old provider (non-fatal): %s", e)

        logger.info("Reloaded %s (full reload complete)", new_config.id)
        return {"status": "ok", "action": "full_reload", "model": new_config.id}


@app.post("/generate")
async def generate(request: Request):
    """Generate text or image depending on model category.

    A ValueError raised anywhere in the provider stack is surfaced as
    HTTP 400 with a structured JSON body the gateway can forward to
    the client verbatim — instead of FastAPI's default 500 "Internal
    Server Error" which hides the root cause.

    Holds `reload_lock` for the duration of inference so a concurrent
    /reload can't yank the provider out from under us.
    """
    async with request.app.state.reload_lock:
        provider: BaseProvider = request.app.state.provider
        config = request.app.state.config
        body = await request.json()

        try:
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
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request"}},
                status_code=400,
            )

        return JSONResponse(
            {"error": {"message": f"Unknown category: {config.category}"}},
            status_code=400,
        )


@app.post("/synthesize")
async def synthesize(request: Request):
    """Synthesize speech. Wrapped in `reload_lock` for the same reason
    /generate is — see its docstring."""
    async with request.app.state.reload_lock:
        provider: BaseProvider = request.app.state.provider
        body = await request.json()

        try:
            text = body.pop("text")
            audio_bytes = await provider.synthesize(text, **body)
            return Response(content=audio_bytes, media_type="application/octet-stream")
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request"}},
                status_code=400,
            )
