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

    # Start unloaded — gateway's VRAM planner calls /load when it's made room.
    logger.info("Worker started (model unloaded): %s — awaiting /load", config.id)
    yield

    logger.info("Worker shutting down: %s", config.id)
    if provider.is_loaded():
        await provider.unload()


app = FastAPI(title="InferGate Worker", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    """Return liveness + model/category metadata."""
    provider: BaseProvider = request.app.state.provider
    config = request.app.state.config
    return {
        "status": "ok" if provider.is_loaded() else "loading",
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
    """Load the configured model; returns structured 503 on provider.load failure."""
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
    """Unload the current model."""
    provider: BaseProvider = request.app.state.provider
    await provider.unload()
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
