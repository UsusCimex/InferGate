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

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.responses import StreamingResponse

from app.config import ModelConfig, load_single_model_config
from app.providers.base import BaseProvider
from app.providers.registry import get_provider_class

logger = logging.getLogger(__name__)


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

    app.state.provider = provider
    app.state.config = config
    app.state.models_dir = models_dir
    # Serialises /reload vs /generate+/synthesize. Brief sequentialisation
    # during reload for max_concurrent>1 models — rare, acceptable.
    app.state.reload_lock = asyncio.Lock()

    # Lazy model load: the gateway calls POST /load on first real request,
    # after its LRU/VRAM-budget planner has made room. Starting "cold" keeps
    # all workers from fighting for VRAM simultaneously on boot.
    logger.info("Worker started (model unloaded): %s — awaiting /load", config.id)
    yield

    logger.info("Worker shutting down: %s", config.id)
    if provider.is_loaded():
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


@app.get("/stats")
async def stats(request: Request):
    """Live resource usage — polled by the gateway's VRAM watchdog every
    few seconds. Keep cheap: no synchronize(), no per-model introspection.
    All fields best-effort — missing tools (no CUDA / no psutil / no
    nvidia-ml-py) give 0/null instead of erroring so the gateway side
    can still reason about whatever is present."""
    provider: BaseProvider = request.app.state.provider
    config = request.app.state.config

    # VRAM reading precedence:
    #   1) NVML (pynvml / nvidia-ml-py) — device-wide usage across ALL
    #      processes on this GPU. Essential for shared-GPU / multi-tenant
    #      deploys where this worker isn't the only torch process.
    #   2) torch.cuda.mem_get_info — per-process caching-allocator view.
    #      Accurate for our own usage but blind to co-tenants.
    # Fall through both ways on ImportError so worker images without
    # nvidia-ml-py still report what they can via torch.
    vram_used_mb = 0
    vram_total_mb = 0
    vram_free_mb = 0
    vram_source = "none"
    try:
        import pynvml  # nvidia-ml-py ships this name
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram_total_mb = info.total // (1024 * 1024)
            vram_used_mb = info.used // (1024 * 1024)
            vram_free_mb = info.free // (1024 * 1024)
            vram_source = "nvml"
        finally:
            pynvml.nvmlShutdown()
    except ImportError:
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info(0)
                vram_free_mb = free_b // (1024 * 1024)
                vram_total_mb = total_b // (1024 * 1024)
                vram_used_mb = vram_total_mb - vram_free_mb
                vram_source = "torch.cuda.mem_get_info"
        except Exception:
            pass
    except Exception:
        # NVML initialised but a later call failed — leave vram_* at 0
        # rather than falling back to torch (NVML failures usually mean
        # driver issues that torch won't work around).
        pass

    ram_used_mb = 0
    ram_total_mb = 0
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_total_mb = vm.total // (1024 * 1024)
        ram_used_mb = (vm.total - vm.available) // (1024 * 1024)
    except ImportError:
        pass
    except Exception:
        pass

    return {
        "model": config.id,
        "loaded": provider.is_loaded(),
        "vram_used_mb": vram_used_mb,
        "vram_free_mb": vram_free_mb,
        "vram_total_mb": vram_total_mb,
        "vram_source": vram_source,
        "ram_used_mb": ram_used_mb,
        "ram_total_mb": ram_total_mb,
        "declared_vram_mb": config.model.get("vram_mb", 0),
    }


@app.post("/load")
async def load(request: Request):
    """Explicit load signal from gateway. Loads model if not loaded.

    Returns 503 with a structured body when the underlying provider.load
    fails — a generic FastAPI 500 with a traceback would be swallowed by
    remote.py's error handler without surfacing the cause.
    """
    provider: BaseProvider = request.app.state.provider
    if provider.is_loaded():
        return {"status": "ok", "model": request.app.state.config.id}

    models_dir = os.environ.get("WORKER_MODELS_DIR", "./models")
    try:
        await provider.load(models_dir)
    except Exception as e:
        logger.error("Load failed for %s: %s", request.app.state.config.id, e)
        return JSONResponse(
            {"error": {"message": str(e), "type": "load_failed"}},
            status_code=503,
        )
    logger.info("Loaded %s via /load", request.app.state.config.id)
    return {"status": "ok", "model": request.app.state.config.id}


@app.post("/unload")
async def unload(request: Request):
    provider: BaseProvider = request.app.state.provider
    await provider.unload()
    return {"status": "ok"}


@app.post("/reload")
async def reload_config(request: Request):
    """Hot-swap the provider from a new ModelConfig posted by the gateway.

    Returns action=noop|metadata|full_reload. Holds reload_lock, which
    also gates /generate and /synthesize, so inflight requests finish
    before the swap. Errors: 400 on bad config / unknown provider_class,
    500 if new provider's load() raises (old provider kept alive).
    """
    body = await request.json()
    try:
        new_config = ModelConfig(**body)
    except Exception as e:
        return JSONResponse(
            {"error": {"message": f"invalid config: {e}", "type": "invalid_request"}},
            status_code=400,
        )

    async with request.app.state.reload_lock:
        old_provider: BaseProvider = request.app.state.provider
        old_config: ModelConfig = request.app.state.config

        # Normalise both configs through JSON-mode model_dump so any
        # pydantic/type-coercion differences (int vs float, list vs tuple
        # for tags, omegaconf ListConfig vs list) collapse to the same
        # representation on both sides. Raw `old_config.model != new.model`
        # failed here because one side had values parsed from YAML via
        # omegaconf while the other came off the wire through pydantic
        # JSON — structurally equal, representationally different.
        old_dump = old_config.model_dump(mode="json")
        new_dump = new_config.model_dump(mode="json")

        if old_dump == new_dump:
            return {"status": "ok", "action": "noop", "model": new_config.id}

        # Worker-observable fields only: `worker_url` is gateway-scope
        # (worker doesn't know its own URL, gateway resolves it from env
        # and ships it in the reload body) so comparing it would spurious-
        # trigger a full reload on every metadata edit.
        full_reload_needed = (
            old_dump.get("provider_class") != new_dump.get("provider_class")
            or old_dump.get("model") != new_dump.get("model")
        )

        if full_reload_needed:
            triggers = []
            if old_dump.get("provider_class") != new_dump.get("provider_class"):
                triggers.append(
                    f"provider_class: {old_dump.get('provider_class')!r} → "
                    f"{new_dump.get('provider_class')!r}"
                )
            if old_dump.get("model") != new_dump.get("model"):
                old_m = old_dump.get("model") or {}
                new_m = new_dump.get("model") or {}
                for k in sorted(set(old_m) | set(new_m)):
                    if old_m.get(k) != new_m.get(k):
                        triggers.append(
                            f"model.{k}: {old_m.get(k)!r} → {new_m.get(k)!r}"
                        )
            logger.info(
                "Full reload of %s triggered by: %s",
                new_config.id, "; ".join(triggers) or "<unknown>",
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
        except Exception as e:
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
        except Exception as e:
            logger.warning("Error unloading old provider (non-fatal): %s", e)

        logger.info("Reloaded %s (full reload complete)", new_config.id)
        return {"status": "ok", "action": "full_reload", "model": new_config.id}


@app.post("/generate")
async def generate(request: Request):
    """Generate text or image. ValueError → HTTP 400 with structured body."""
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
    """Synthesize speech."""
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


@app.post("/voice-clone")
async def voice_clone(
    request: Request,
    reference_audio: UploadFile = File(...),
    input: str = Form(...),
    reference_text: str | None = Form(None),
    speed: float = Form(1.0),
    output_format: str = Form("mp3"),
):
    """Voice-cloning TTS — multipart because reference is a raw clip."""
    async with request.app.state.reload_lock:
        provider: BaseProvider = request.app.state.provider
        ref = await reference_audio.read()
        params: dict = {
            "speed": speed,
            "output_format": output_format,
            "reference_audio": ref,
            "reference_filename": reference_audio.filename or "ref.wav",
        }
        if reference_text is not None:
            params["reference_text"] = reference_text
        try:
            audio_bytes = await provider.synthesize(input, **params)
            return Response(content=audio_bytes, media_type="application/octet-stream")
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request"}},
                status_code=400,
            )


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """Transcribe audio → text. Multipart form-data to mirror /v1/audio/transcriptions."""
    async with request.app.state.reload_lock:
        provider: BaseProvider = request.app.state.provider
        audio = await file.read()
        params = {
            "language": language,
            "prompt": prompt,
            "response_format": response_format,
            "temperature": temperature,
            "filename": file.filename or "audio.wav",
        }
        params = {k: v for k, v in params.items() if v is not None}
        try:
            result = await provider.transcribe(audio, **params)
            return JSONResponse(result)
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request"}},
                status_code=400,
            )


@app.post("/upscale")
async def upscale(request: Request, file: UploadFile = File(...)):
    """Super-resolution endpoint. Image in, image out (raw PNG bytes)."""
    async with request.app.state.reload_lock:
        provider: BaseProvider = request.app.state.provider
        image = await file.read()
        try:
            png_bytes = await provider.upscale(image)
            return Response(content=png_bytes, media_type="image/png")
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request"}},
                status_code=400,
            )
