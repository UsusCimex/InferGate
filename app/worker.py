from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.responses import StreamingResponse

from app.config import ModelConfig, load_single_model_config
from app.providers.base import BaseProvider
from app.providers.registry import get_provider_class

logger = logging.getLogger(__name__)

_UNLOAD_CANCEL_TIMEOUT = 10.0


def _initial_load_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "error": None,
        "started_at": None,
        "duration_seconds": None,
        "cancellation_requested": False,
    }


async def _run_load(app: FastAPI, models_dir: str) -> None:
    """Background load task — owns load_lock; updates load_state on completion/failure."""
    state: dict[str, Any] = app.state.load_state
    provider: BaseProvider = app.state.provider
    started = time.monotonic()
    state["started_at"] = started
    state["duration_seconds"] = None
    state["error"] = None
    state["status"] = "loading"

    async with app.state.load_lock:
        try:
            await provider.load(models_dir)
            state["status"] = "ready"
            logger.info(
                "Loaded %s in %.1fs (background)",
                app.state.config.id, time.monotonic() - started,
            )
        except asyncio.CancelledError:
            state["status"] = "idle"
            state["error"] = "cancelled"
            logger.info("Background load of %s cancelled", app.state.config.id)
            raise
        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            logger.error("Background load of %s failed: %s", app.state.config.id, e)
        finally:
            state["duration_seconds"] = time.monotonic() - started


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
    # Serialises /reload vs /generate+/synthesize.
    app.state.reload_lock = asyncio.Lock()
    # Held by the background /load task across provider.load(); also acquired by /reload
    # (reload_lock first → load_lock) and /unload to serialise model swaps.
    app.state.load_lock = asyncio.Lock()
    app.state.load_state = _initial_load_state()
    app.state.load_task = None

    # Start unloaded — gateway's VRAM planner calls /load when it's made room.
    logger.info("Worker started (model unloaded): %s — awaiting /load", config.id)
    yield

    logger.info("Worker shutting down: %s", config.id)
    task: asyncio.Task | None = app.state.load_task
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(task, timeout=_UNLOAD_CANCEL_TIMEOUT)
    if provider.is_loaded():
        await provider.unload()


app = FastAPI(title="InferGate Worker", lifespan=lifespan)


_INFERENCE_PATHS = frozenset({
    "/generate",
    "/synthesize",
    "/voice-clone",
    "/transcribe",
    "/upscale",
    "/embed",
    "/embed-audio",
    "/embed-image",
    "/embed-video",
})


@app.middleware("http")
async def _ready_guard(request: Request, call_next):
    """Return 503 from inference endpoints when the model isn't yet ready."""
    if request.url.path in _INFERENCE_PATHS:
        state = getattr(request.app.state, "load_state", None)
        if state is not None and state.get("status") != "ready":
            return JSONResponse(
                {"error": {
                    "message": f"model not ready (status={state.get('status')})",
                    "type": "model_not_ready",
                }},
                status_code=503,
            )
    return await call_next(request)


@app.get("/health")
async def health(request: Request):
    """Return liveness + model/category metadata."""
    state = getattr(request.app.state, "load_state", None)
    config = request.app.state.config
    if state is not None:
        ready = state.get("status") == "ready"
    else:
        # Fixture-built apps may not init load_state — fall back to provider state.
        ready = request.app.state.provider.is_loaded()
    return {
        "status": "ok" if ready else "loading",
        "model": config.id,
        "category": config.category,
    }


@app.get("/stats")
async def stats(request: Request):
    """Return live VRAM + host-RAM usage for the watchdog."""
    provider: BaseProvider = request.app.state.provider
    config = request.app.state.config

    # VRAM source priority: NVML (device-wide, sees co-tenants) → torch (per-process).
    vram_used_mb = 0
    vram_total_mb = 0
    vram_free_mb = 0
    vram_source = "none"
    try:
        import pynvml
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
        # NVML post-init failure usually means driver issues that torch won't recover from.
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
    """Start the configured model loading; returns 200 if ready, 202 if loading begins."""
    app_state = request.app.state
    config = app_state.config

    # Fixture-built apps without load_state still want sync semantics for legacy tests.
    if not hasattr(app_state, "load_state"):
        provider: BaseProvider = app_state.provider
        if provider.is_loaded():
            return {"status": "ok", "model": config.id}
        models_dir = os.environ.get("WORKER_MODELS_DIR", "./models")
        try:
            await provider.load(models_dir)
        except Exception as e:
            logger.error("Load failed for %s: %s", config.id, e)
            return JSONResponse(
                {"error": {"message": str(e), "type": "load_failed"}},
                status_code=503,
            )
        return {"status": "ok", "model": config.id}

    state: dict[str, Any] = app_state.load_state
    status = state["status"]

    if status == "ready":
        return {"status": "ok", "model": config.id}

    if status in ("loading", "cancelling"):
        return JSONResponse(
            {"status": "loading", "model": config.id, "load_state": _public_state(state)},
            status_code=202,
        )

    # Reap any prior task before starting a new one — avoids parallel loads.
    prior: asyncio.Task | None = app_state.load_task
    if prior is not None and not prior.done():
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(prior, timeout=_UNLOAD_CANCEL_TIMEOUT)

    state.update(_initial_load_state())
    state["status"] = "loading"
    state["started_at"] = time.monotonic()
    models_dir = os.environ.get("WORKER_MODELS_DIR", app_state.models_dir)
    app_state.load_task = asyncio.create_task(_run_load(request.app, models_dir))
    return JSONResponse(
        {"status": "loading", "model": config.id, "load_state": _public_state(state)},
        status_code=202,
    )


@app.get("/load/status")
async def load_status(request: Request):
    """Return the current background-load state machine snapshot."""
    state = getattr(request.app.state, "load_state", None)
    if state is None:
        # Legacy fixture path — synthesise a stable shape from provider.is_loaded().
        loaded = request.app.state.provider.is_loaded()
        return {
            "status": "ready" if loaded else "idle",
            "model": request.app.state.config.id,
        }
    return {"model": request.app.state.config.id, **_public_state(state)}


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    """Strip internal fields before returning state to the gateway."""
    return {
        "status": state["status"],
        "error": state.get("error"),
        "started_at": state.get("started_at"),
        "duration_seconds": state.get("duration_seconds"),
    }


@app.post("/unload")
async def unload(request: Request):
    """Cancel any in-flight load, then unload the model."""
    app_state = request.app.state
    state = getattr(app_state, "load_state", None)
    task: asyncio.Task | None = getattr(app_state, "load_task", None)

    if state is not None and task is not None and not task.done():
        state["cancellation_requested"] = True
        state["status"] = "cancelling"
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=_UNLOAD_CANCEL_TIMEOUT)
        except (asyncio.CancelledError, TimeoutError):
            return JSONResponse(
                {"status": "cancelling", "note": "load may continue in background"},
                status_code=409,
            )

    # load_lock guarantees we don't race a freshly started load.
    lock = getattr(app_state, "load_lock", None)
    provider: BaseProvider = app_state.provider
    if lock is not None:
        async with lock:
            await provider.unload()
    else:
        await provider.unload()

    if state is not None:
        state.update(_initial_load_state())
    return {"status": "ok"}


@app.post("/reload")
async def reload_config(request: Request):
    """Hot-swap the provider from a new ModelConfig (returns action=noop|metadata|full_reload)."""
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

        # Compare via JSON-mode dumps so omegaconf vs pydantic representations collapse.
        old_dump = old_config.model_dump(mode="json")
        new_dump = new_config.model_dump(mode="json")

        if old_dump == new_dump:
            return {"status": "ok", "action": "noop", "model": new_config.id}

        # worker_url is gateway-scope and must be excluded from the diff.
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
            old_provider.config = new_config
            logger.info("Reloaded %s (metadata only)", new_config.id)
            return {"status": "ok", "action": "metadata", "model": new_config.id}

        # Load new before touching old so worker stays serviceable on bad config.
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
    """Generate text or image; ValueError from the provider maps to HTTP 400."""
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
    """Synthesise speech from a JSON text payload."""
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
    """Synthesise speech in the voice of `reference_audio` (multipart upload)."""
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
    """Transcribe audio to text."""
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
    """Super-resolve an uploaded image; returns raw PNG bytes."""
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


@app.post("/embed")
async def embed(request: Request):
    """Encode a batch of texts; returns {'embeddings': list[list[float]]}."""
    async with request.app.state.reload_lock:
        provider: BaseProvider = request.app.state.provider
        body = await request.json()
        try:
            inputs = body.pop("input")
            if isinstance(inputs, str):
                inputs = [inputs]
            vecs = await provider.embed(inputs, **body)
            return JSONResponse({"embeddings": vecs})
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request"}},
                status_code=400,
            )


@app.post("/embed-audio")
async def embed_audio(request: Request, file: UploadFile = File(...)):
    """Encode an audio chunk; returns {'embedding': list[float]}."""
    async with request.app.state.reload_lock:
        provider: BaseProvider = request.app.state.provider
        audio = await file.read()
        try:
            vec = await provider.embed(audio, filename=file.filename or "audio.wav")
            return JSONResponse({"embedding": vec})
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request"}},
                status_code=400,
            )


@app.post("/embed-image")
async def embed_image(request: Request, file: UploadFile = File(...)):
    """Encode an image; returns {'embedding': list[float]}."""
    async with request.app.state.reload_lock:
        provider: BaseProvider = request.app.state.provider
        image = await file.read()
        try:
            vec = await provider.embed_image(image, filename=file.filename or "image.jpg")
            return JSONResponse({"embedding": vec})
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request"}},
                status_code=400,
            )


@app.post("/embed-video")
async def embed_video(request: Request, file: UploadFile = File(...)):
    """Encode a video clip; returns {'embedding': list[float]}."""
    async with request.app.state.reload_lock:
        provider: BaseProvider = request.app.state.provider
        video = await file.read()
        try:
            vec = await provider.embed_video(video, filename=file.filename or "clip.mp4")
            return JSONResponse({"embedding": vec})
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request"}},
                status_code=400,
            )
