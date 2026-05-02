from __future__ import annotations

import hashlib
import time

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.dependencies import (
    get_cache_manager,
    get_defaults,
    get_gpu_scheduler,
    get_provider_manager,
    get_upload_limits,
)
from app.monitoring import CACHE_HITS, CACHE_MISSES, INFERENCE_DURATION, is_prometheus_available
from app.schemas.audio import AudioSpeechRequest
from app.utils import read_with_limit

router = APIRouter()

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
}


@router.post("/v1/audio/speech")
async def create_speech(
    body: AudioSpeechRequest,
    request: Request,
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    cache=Depends(get_cache_manager),
    defaults=Depends(get_defaults),
):
    model_id = body.model or defaults.get("tts")
    if not model_id:
        return JSONResponse({"error": {"message": "No model specified"}}, status_code=400)

    config = manager.get_config(model_id)
    if config.capabilities.voice_clone_only:
        return JSONResponse(
            {
                "error": {
                    "message": (
                        f"model '{model_id}' is voice-clone-only and requires a reference_audio "
                        f"clip — call POST /v1/audio/speech/voice-clone (multipart) instead"
                    ),
                    "type": "voice_clone_required",
                    "endpoint": "/v1/audio/speech/voice-clone",
                }
            },
            status_code=400,
        )

    start = time.monotonic()
    provider = await manager.ensure_loaded(model_id)

    params = {
        "voice": body.voice,
        "speed": body.speed,
        "output_format": body.response_format,
    }
    if body.language is not None:
        params["language"] = body.language

    # Cache check
    no_cache = request.headers.get("X-InferGate-No-Cache", "").lower() == "true"
    cache_cfg = config.cache.model_dump()
    should_cache = not no_cache and cache.should_cache(cache_cfg, params)
    cache_key = cache.make_key(model_id, {"input": body.input, **params})
    cache_status = "DISABLED"

    if should_cache:
        cached = await cache.get(cache_key)
        if cached:
            elapsed = int((time.monotonic() - start) * 1000)
            content_type = CONTENT_TYPES.get(body.response_format, "application/octet-stream")
            if is_prometheus_available():
                CACHE_HITS.labels(model_id=model_id).inc()
            return Response(
                content=cached,
                media_type=content_type,
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

    # Generate
    timeout = config.queue.timeout_seconds
    priority = config.queue.priority

    inference_start = time.monotonic()
    async with manager.active_request(model_id):
        audio_bytes = await scheduler.submit(
            model_id, priority, provider.synthesize(body.input, **params), timeout
        )
    if is_prometheus_available():
        INFERENCE_DURATION.labels(model_id=model_id, category="tts").observe(
            time.monotonic() - inference_start
        )

    if should_cache:
        await cache.put(cache_key, audio_bytes, model_id, cache_cfg)

    elapsed = int((time.monotonic() - start) * 1000)
    content_type = CONTENT_TYPES.get(body.response_format, "application/octet-stream")
    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "X-InferGate-Cache": cache_status,
            "X-InferGate-Model": model_id,
            "X-InferGate-Queue-Position": str(scheduler.last_position),
            "X-InferGate-Generation-Ms": str(elapsed),
        },
    )


_TRANSCRIPTION_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds as the SRT timestamp `HH:MM:SS,ms`."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = round((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_time(seconds: float) -> str:
    """Format seconds as the WebVTT timestamp `HH:MM:SS.ms`."""
    return _fmt_srt_time(seconds).replace(",", ".")


def _segments_to_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}")
        lines.append(seg.get("text", "").strip())
        lines.append("")
    return "\n".join(lines)


def _segments_to_vtt(segments: list[dict]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_fmt_vtt_time(seg['start'])} --> {_fmt_vtt_time(seg['end'])}")
        lines.append(seg.get("text", "").strip())
        lines.append("")
    return "\n".join(lines)


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    cache=Depends(get_cache_manager),
    defaults=Depends(get_defaults),
    limits=Depends(get_upload_limits),
):
    """OpenAI-compatible multipart transcription endpoint."""
    if response_format not in _TRANSCRIPTION_FORMATS:
        return JSONResponse(
            {"error": {"message": f"response_format must be one of {sorted(_TRANSCRIPTION_FORMATS)}"}},
            status_code=400,
        )

    # Subtitle formats are rendered from provider verbose_json — keeps providers focused.
    effective_format = "verbose_json" if response_format in {"srt", "vtt"} else response_format

    model_id = model or defaults.get("stt")
    if not model_id:
        return JSONResponse({"error": {"message": "No STT model specified"}}, status_code=400)

    audio_bytes = await read_with_limit(file, limits.max_audio_mb * 1024 * 1024)
    if not audio_bytes:
        return JSONResponse({"error": {"message": "Empty audio file"}}, status_code=400)

    start = time.monotonic()
    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    params = {
        "language": language,
        "prompt": prompt,
        "response_format": effective_format,
        "temperature": temperature,
        "filename": file.filename or "audio.wav",
    }
    params = {k: v for k, v in params.items() if v is not None}

    no_cache = request.headers.get("X-InferGate-No-Cache", "").lower() == "true"
    cache_cfg = config.cache.model_dump()
    should_cache = not no_cache and cache.should_cache(cache_cfg, params)
    cache_key = cache.make_key(model_id, {"audio_len": len(audio_bytes), "audio_sha": _sha(audio_bytes), **params})
    cache_status = "DISABLED"

    if should_cache:
        cached = await cache.get(cache_key)
        if cached:
            elapsed = int((time.monotonic() - start) * 1000)
            if is_prometheus_available():
                CACHE_HITS.labels(model_id=model_id).inc()
            return _transcription_response(
                cached, response_format, model_id, elapsed, "HIT", scheduler.last_position
            )
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
        result = await scheduler.submit(
            model_id, priority, provider.transcribe(audio_bytes, **params), timeout
        )
    if is_prometheus_available():
        INFERENCE_DURATION.labels(model_id=model_id, category="stt").observe(
            time.monotonic() - inference_start
        )

    if should_cache:
        import json as _json
        await cache.put(cache_key, _json.dumps(result).encode(), model_id, cache_cfg)

    elapsed = int((time.monotonic() - start) * 1000)
    return _transcription_response(
        result, response_format, model_id, elapsed, cache_status, scheduler.last_position
    )


def _sha(data: bytes) -> str:
    """Return a short (32-hex) content hash for cache keying."""
    import hashlib
    return hashlib.sha256(data).hexdigest()[:32]


def _transcription_response(
    result, response_format: str, model_id: str, elapsed_ms: int,
    cache_status: str, queue_position: int,
):
    """Format provider output as the requested `response_format` (json/text/srt/vtt/verbose_json)."""
    import json as _json
    if isinstance(result, (bytes, bytearray)):
        result = _json.loads(result.decode())

    headers = {
        "X-InferGate-Cache": cache_status,
        "X-InferGate-Model": model_id,
        "X-InferGate-Queue-Position": str(queue_position),
        "X-InferGate-Generation-Ms": str(elapsed_ms),
    }

    if response_format == "text":
        return Response(content=result.get("text", ""), media_type="text/plain", headers=headers)
    if response_format == "verbose_json":
        return JSONResponse(result, headers=headers)
    if response_format == "srt":
        body = _segments_to_srt(result.get("segments") or [])
        return Response(content=body, media_type="application/x-subrip", headers=headers)
    if response_format == "vtt":
        body = _segments_to_vtt(result.get("segments") or [])
        return Response(content=body, media_type="text/vtt", headers=headers)
    # Narrow to {text: ...} for strict OpenAI parity.
    return JSONResponse({"text": result.get("text", "")}, headers=headers)


@router.post("/v1/audio/speech/voice-clone")
async def create_speech_voice_clone(
    request: Request,
    reference_audio: UploadFile = File(...),
    input: str = Form(..., min_length=1, max_length=100000),
    model: str | None = Form(None),
    reference_text: str | None = Form(None),
    response_format: str = Form("mp3"),
    speed: float = Form(1.0, ge=0.25, le=4.0),
    language: str | None = Form(None, max_length=32),
    manager=Depends(get_provider_manager),
    scheduler=Depends(get_gpu_scheduler),
    cache=Depends(get_cache_manager),
    defaults=Depends(get_defaults),
    limits=Depends(get_upload_limits),
):
    """Synthesise `input` in the voice from `reference_audio` (multipart upload)."""
    model_id = model or defaults.get("tts")
    if not model_id:
        return JSONResponse({"error": {"message": "No model specified"}}, status_code=400)

    ref_bytes = await read_with_limit(reference_audio, limits.max_audio_mb * 1024 * 1024)
    if not ref_bytes:
        return JSONResponse({"error": {"message": "Empty reference_audio"}}, status_code=400)

    start = time.monotonic()
    provider = await manager.ensure_loaded(model_id)
    config = manager.get_config(model_id)

    params = {
        "speed": speed,
        "output_format": response_format,
        "reference_audio": ref_bytes,
        "reference_filename": reference_audio.filename or "ref.wav",
        "reference_text": reference_text,
        "language": language,
    }
    params = {k: v for k, v in params.items() if v is not None}

    no_cache = request.headers.get("X-InferGate-No-Cache", "").lower() == "true"
    cache_cfg = config.cache.model_dump()
    ref_sha = hashlib.sha256(ref_bytes).hexdigest()[:32]
    cache_params = {
        "input": input, "ref_sha": ref_sha, "reference_text": reference_text or "",
        "speed": speed, "response_format": response_format,
        "language": language or "",
    }
    should_cache = not no_cache and cache.should_cache(cache_cfg, cache_params)
    cache_key = cache.make_key(model_id, cache_params)
    cache_status = "DISABLED"

    content_type = CONTENT_TYPES.get(response_format, "application/octet-stream")

    if should_cache:
        cached = await cache.get(cache_key)
        if cached:
            elapsed = int((time.monotonic() - start) * 1000)
            if is_prometheus_available():
                CACHE_HITS.labels(model_id=model_id).inc()
            return Response(
                content=cached, media_type=content_type,
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

    inference_start = time.monotonic()
    async with manager.active_request(model_id):
        audio_bytes = await scheduler.submit(
            model_id, priority, provider.synthesize(input, **params), timeout
        )
    if is_prometheus_available():
        INFERENCE_DURATION.labels(model_id=model_id, category="tts").observe(
            time.monotonic() - inference_start
        )

    if should_cache:
        await cache.put(cache_key, audio_bytes, model_id, cache_cfg)

    elapsed = int((time.monotonic() - start) * 1000)
    return Response(
        content=audio_bytes, media_type=content_type,
        headers={
            "X-InferGate-Cache": cache_status,
            "X-InferGate-Model": model_id,
            "X-InferGate-Queue-Position": str(scheduler.last_position),
            "X-InferGate-Generation-Ms": str(elapsed),
        },
    )
