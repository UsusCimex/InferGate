"""End-to-end tests: RemoteProvider ⇄ FakeWorker via httpx ASGITransport.

These tests boot a FastAPI app that mimics worker.py's contract and route
RemoteProvider's httpx client through ASGITransport — no sockets, no mocks,
the full HTTP serialization round-trip.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from httpx import ASGITransport

from app.config import ModelCacheConfig, ModelConfig, ModelMetadata, ModelQueueConfig
from app.providers import remote as remote_module


def _make_remote_config(
    category: str, model_id: str | None = None
) -> ModelConfig:
    return ModelConfig(
        id=model_id or f"e2e-{category}",
        display_name=f"E2E {category}",
        category=category,
        provider_class="Unused",
        enabled=True,
        worker_url="http://fake-worker",
        model={"hub_id": "test/test", "vram_mb": 0},
        cache=ModelCacheConfig(),
        queue=ModelQueueConfig(),
        metadata=ModelMetadata(),
    )


def _make_fake_worker() -> tuple[FastAPI, dict[str, Any]]:
    """Build a FastAPI app that records every request and replays canned responses."""
    app = FastAPI()
    state: dict[str, Any] = {
        "loaded": False,
        "load_calls": 0,
        "unload_calls": 0,
        "last_request_id": None,
        "last_payload": None,
        "last_form": None,
        "last_file_size": 0,
    }

    @app.middleware("http")
    async def capture_request_id(request: Request, call_next):
        state["last_request_id"] = request.headers.get("x-request-id")
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok" if state["loaded"] else "loading"}

    @app.post("/load")
    async def load():
        state["loaded"] = True
        state["load_calls"] += 1
        return {"status": "ok"}

    @app.post("/unload")
    async def unload():
        state["loaded"] = False
        state["unload_calls"] += 1
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        state["last_payload"] = body
        if "messages" in body:
            return JSONResponse({
                "id": "chat-1",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "echo:" + body["messages"][-1]["content"],
                    },
                    "finish_reason": "stop",
                }],
            })
        if "prompt" in body:
            return Response(content=b"\x89PNG-FAKE-" + body["prompt"].encode(),
                            media_type="image/png")
        return JSONResponse({"error": "unsupported"}, status_code=400)

    @app.post("/synthesize")
    async def synthesize(request: Request):
        body = await request.json()
        state["last_payload"] = body
        return Response(content=b"WAV:" + body["text"].encode(),
                        media_type="application/octet-stream")

    @app.post("/voice-clone")
    async def voice_clone(
        reference_audio: UploadFile = File(...),
        input: str = Form(...),
        speed: str = Form("1.0"),
    ):
        ref = await reference_audio.read()
        state["last_form"] = {"input": input, "speed": speed}
        state["last_file_size"] = len(ref)
        return Response(content=b"CLONE:" + input.encode() + b"|" + ref,
                        media_type="application/octet-stream")

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
    ):
        audio = await file.read()
        state["last_file_size"] = len(audio)
        state["last_form"] = {"language": language, "prompt": prompt}
        return JSONResponse({
            "text": f"transcribed-{len(audio)}-bytes",
            "language": language or "en",
        })

    @app.post("/upscale")
    async def upscale(file: UploadFile = File(...)):
        image = await file.read()
        state["last_file_size"] = len(image)
        return Response(content=b"UP:" + image, media_type="image/png")

    @app.post("/embed")
    async def embed(request: Request):
        body = await request.json()
        state["last_payload"] = body
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        return JSONResponse({
            "embeddings": [[float(len(s)), 0.0, 1.0] for s in inputs],
        })

    @app.post("/embed-audio")
    async def embed_audio(file: UploadFile = File(...)):
        audio = await file.read()
        state["last_file_size"] = len(audio)
        return JSONResponse({"embedding": [float(len(audio)), 0.0, 1.0]})

    @app.post("/embed-image")
    async def embed_image(file: UploadFile = File(...)):
        image = await file.read()
        state["last_file_size"] = len(image)
        return JSONResponse({"embedding": [float(len(image)), 1.0, 0.0]})

    @app.post("/embed-video")
    async def embed_video(file: UploadFile = File(...)):
        video = await file.read()
        state["last_file_size"] = len(video)
        return JSONResponse({"embedding": [float(len(video)), 0.5, 0.5]})

    return app, state


@pytest_asyncio.fixture
async def fake_worker(monkeypatch):
    """Patch BaseRemoteMixin._build_client to route through ASGITransport."""
    app, state = _make_fake_worker()
    transport = ASGITransport(app=app)

    def _build_with_transport(self, timeout):
        return httpx.AsyncClient(
            base_url=self._worker_url,
            timeout=timeout,
            transport=transport,
        )

    monkeypatch.setattr(
        remote_module.BaseRemoteMixin, "_build_client", _build_with_transport
    )
    yield state


# ── Lifecycle ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_load_unload_cycle(fake_worker):
    provider = remote_module.RemoteTextProvider(_make_remote_config("text"))
    assert not provider.is_loaded()

    # Worker reports "loading" until /load is called → BaseRemoteMixin tolerates that.
    fake_worker["loaded"] = True  # /health returns ok
    await provider.load("/tmp")
    assert provider.is_loaded()
    assert fake_worker["load_calls"] == 1

    await provider.unload()
    assert not provider.is_loaded()
    assert fake_worker["unload_calls"] == 1


@pytest.mark.asyncio
async def test_e2e_load_fails_when_health_unreachable(monkeypatch):
    """If /health returns 500, load() must surface RuntimeError and not mark loaded."""
    bad_app = FastAPI()

    @bad_app.get("/health")
    async def _bad_health():
        return JSONResponse({"status": "down"}, status_code=500)

    bad_transport = ASGITransport(app=bad_app)

    def _build(self, timeout):
        return httpx.AsyncClient(
            base_url=self._worker_url, timeout=timeout, transport=bad_transport
        )

    monkeypatch.setattr(remote_module.BaseRemoteMixin, "_build_client", _build)

    provider = remote_module.RemoteTextProvider(_make_remote_config("text"))
    with pytest.raises(RuntimeError, match="returned status 500"):
        await provider.load("/tmp")
    assert not provider.is_loaded()


# ── JSON endpoints ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_text_generate(fake_worker):
    fake_worker["loaded"] = True
    provider = remote_module.RemoteTextProvider(_make_remote_config("text"))
    await provider.load("/tmp")

    result = await provider.generate(
        [{"role": "user", "content": "hello"}],
        temperature=0.7,
    )
    assert result["choices"][0]["message"]["content"] == "echo:hello"
    assert fake_worker["last_payload"]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_e2e_image_generate_returns_bytes(fake_worker):
    fake_worker["loaded"] = True
    provider = remote_module.RemoteImageProvider(_make_remote_config("image"))
    await provider.load("/tmp")

    png = await provider.generate("a red cat", size="512x512")
    assert png.startswith(b"\x89PNG-FAKE-")
    assert b"a red cat" in png
    assert fake_worker["last_payload"]["size"] == "512x512"


@pytest.mark.asyncio
async def test_e2e_tts_synthesize_json_path(fake_worker):
    fake_worker["loaded"] = True
    provider = remote_module.RemoteTtsProvider(_make_remote_config("tts"))
    await provider.load("/tmp")

    audio = await provider.synthesize("hi there", voice="alex")
    assert audio == b"WAV:hi there"
    assert fake_worker["last_payload"]["voice"] == "alex"


@pytest.mark.asyncio
async def test_e2e_tts_voice_clone_path(fake_worker):
    fake_worker["loaded"] = True
    provider = remote_module.RemoteTtsProvider(_make_remote_config("tts"))
    await provider.load("/tmp")

    audio = await provider.synthesize(
        "say this",
        reference_audio=b"REF-AUDIO-BYTES",
        reference_filename="ref.wav",
        speed=1.5,
    )
    assert audio.startswith(b"CLONE:say this|REF-AUDIO-BYTES")
    assert fake_worker["last_form"]["speed"] == "1.5"
    assert fake_worker["last_file_size"] == len(b"REF-AUDIO-BYTES")


@pytest.mark.asyncio
async def test_e2e_text_embedding(fake_worker):
    fake_worker["loaded"] = True
    provider = remote_module.RemoteTextEmbeddingProvider(
        _make_remote_config("embedding-text")
    )
    await provider.load("/tmp")

    vecs = await provider.embed(["hi", "hello world"])
    assert len(vecs) == 2
    assert vecs[0][0] == 2.0  # len("hi")
    assert vecs[1][0] == 11.0  # len("hello world")


# ── Multipart endpoints ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_stt_transcribe(fake_worker):
    fake_worker["loaded"] = True
    provider = remote_module.RemoteSttProvider(_make_remote_config("stt"))
    await provider.load("/tmp")

    audio = b"\x00\x01\x02\x03" * 100
    result = await provider.transcribe(audio, filename="a.wav", language="en")
    assert result["text"] == f"transcribed-{len(audio)}-bytes"
    assert result["language"] == "en"
    assert fake_worker["last_form"]["language"] == "en"


@pytest.mark.asyncio
async def test_e2e_upscale(fake_worker):
    fake_worker["loaded"] = True
    provider = remote_module.RemoteUpscaleProvider(_make_remote_config("upscale"))
    await provider.load("/tmp")

    out = await provider.upscale(b"PNG-INPUT")
    assert out == b"UP:PNG-INPUT"
    assert fake_worker["last_file_size"] == len(b"PNG-INPUT")


@pytest.mark.asyncio
async def test_e2e_audio_embedding(fake_worker):
    fake_worker["loaded"] = True
    provider = remote_module.RemoteAudioEmbeddingProvider(
        _make_remote_config("embedding-audio")
    )
    await provider.load("/tmp")

    vec = await provider.embed(b"audio-bytes-here")
    assert vec[0] == float(len(b"audio-bytes-here"))


@pytest.mark.asyncio
async def test_e2e_multimodal_embedding_text_and_image(fake_worker):
    fake_worker["loaded"] = True
    provider = remote_module.RemoteMultimodalEmbeddingProvider(
        _make_remote_config("embedding-multimodal")
    )
    await provider.load("/tmp")

    text_vec = await provider.embed(["hello"])
    assert text_vec[0][0] == 5.0

    img_vec = await provider.embed_image(b"fake-jpeg-bytes")
    assert img_vec[0] == float(len(b"fake-jpeg-bytes"))


@pytest.mark.asyncio
async def test_e2e_video_embedding(fake_worker):
    fake_worker["loaded"] = True
    provider = remote_module.RemoteVideoEmbeddingProvider(
        _make_remote_config("embedding-video")
    )
    await provider.load("/tmp")

    text_vec = await provider.embed(["clip"])
    assert text_vec[0][0] == 4.0

    vid_vec = await provider.embed_video(b"mp4-bytes-payload")
    assert vid_vec[0] == float(len(b"mp4-bytes-payload"))


# ── Cross-cutting concerns ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_request_id_propagated_to_worker(fake_worker):
    """X-Request-ID set in gateway must reach the worker on every call."""
    from app.monitoring import set_request_id

    fake_worker["loaded"] = True
    provider = remote_module.RemoteTextProvider(_make_remote_config("text"))
    await provider.load("/tmp")

    set_request_id("req-trace-xyz")
    try:
        await provider.generate([{"role": "user", "content": "hi"}])
        assert fake_worker["last_request_id"] == "req-trace-xyz"
    finally:
        set_request_id(None)


@pytest.mark.asyncio
async def test_e2e_streaming_text(monkeypatch):
    """Build a worker that emits SSE-style chunks; verify provider yields them."""
    from fastapi.responses import StreamingResponse

    streaming_app = FastAPI()

    @streaming_app.get("/health")
    async def _health():
        return {"status": "ok"}

    @streaming_app.post("/load")
    async def _load():
        return {"status": "ok"}

    @streaming_app.post("/generate")
    async def _stream_generate(request: Request):
        body = await request.json()
        if not body.get("stream"):
            return JSONResponse({"err": "expected stream"}, status_code=400)

        async def _gen():
            yield b"data: chunk-1\n"
            yield b"data: chunk-2\n"
            yield b"data: [DONE]\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    transport = ASGITransport(app=streaming_app)

    def _build(self, timeout):
        return httpx.AsyncClient(
            base_url=self._worker_url, timeout=timeout, transport=transport
        )

    monkeypatch.setattr(remote_module.BaseRemoteMixin, "_build_client", _build)

    provider = remote_module.RemoteTextProvider(_make_remote_config("text"))
    await provider.load("/tmp")
    out_lines: list[str] = []
    async for line in provider.generate_stream([{"role": "user", "content": "hi"}]):
        out_lines.append(line.strip())

    assert "data: chunk-1" in out_lines
    assert "data: chunk-2" in out_lines
    assert "data: [DONE]" in out_lines


@pytest.mark.asyncio
async def test_e2e_load_after_unload_reconnects(fake_worker):
    """A provider can be cycled load → unload → load without errors."""
    fake_worker["loaded"] = True
    provider = remote_module.RemoteTextProvider(_make_remote_config("text"))

    await provider.load("/tmp")
    assert provider.is_loaded()
    await provider.unload()
    assert not provider.is_loaded()

    await provider.load("/tmp")
    assert provider.is_loaded()
    assert fake_worker["load_calls"] == 2
    assert fake_worker["unload_calls"] == 1


@pytest.mark.asyncio
async def test_e2e_check_health_routes_correctly(fake_worker, monkeypatch):
    """check_health() builds its own client; route that one too."""
    state = fake_worker
    state["loaded"] = True

    healthy_app, _ = _make_fake_worker()

    @healthy_app.get("/health")
    async def _ok_health():  # type: ignore[no-redef]
        return {"status": "ok"}

    transport = ASGITransport(app=healthy_app)
    real_AsyncClient = httpx.AsyncClient

    def _patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_AsyncClient(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _patched_client)

    provider = remote_module.RemoteTextProvider(_make_remote_config("text"))
    assert await provider.check_health() is True


# ── Registry ───────────────────────────────────────────────────────


def test_registry_exposes_every_category():
    """remote_provider_for must resolve every category we register."""
    from app.providers.remote import CATEGORY_REGISTRY, remote_provider_for

    assert set(CATEGORY_REGISTRY.keys()) == {
        "text", "image", "tts", "stt", "upscale",
        "embedding-text", "embedding-audio",
        "embedding-multimodal", "embedding-video",
    }
    for cat in CATEGORY_REGISTRY:
        cls = remote_provider_for(cat)
        assert cls is CATEGORY_REGISTRY[cat]


def test_registry_rejects_unknown_category():
    from app.providers.remote import remote_provider_for

    with pytest.raises(ValueError, match="No remote provider"):
        remote_provider_for("clairvoyance")


# ── Async /load polling ─────────────────────────────────────────────


def _make_polling_worker(
    statuses: list[dict],
    *,
    accept_load: bool = True,
) -> FastAPI:
    """Worker that returns 202 on /load and replays `statuses` on each /load/status call."""
    app = FastAPI()
    cursor = {"i": 0}

    @app.get("/health")
    async def _h():
        return {"status": "ok"}

    @app.post("/load")
    async def _l():
        if not accept_load:
            return JSONResponse({"status": "error"}, status_code=500)
        return JSONResponse(
            {"status": "loading", "load_state": {"status": "loading"}},
            status_code=202,
        )

    @app.get("/load/status")
    async def _s():
        i = min(cursor["i"], len(statuses) - 1)
        cursor["i"] += 1
        return statuses[i]

    return app


@pytest.mark.asyncio
async def test_e2e_async_load_polling_succeeds(monkeypatch):
    """FakeWorker reports loading→loading→ready; gateway completes load via polling."""
    from app.providers import remote as r

    monkeypatch.setattr(r, "_LOAD_POLL_BACKOFF", [0.0, 0.0, 0.0])

    app = _make_polling_worker([
        {"status": "loading"},
        {"status": "loading"},
        {"status": "ready"},
    ])
    transport = ASGITransport(app=app)

    def _build(self, timeout):
        return httpx.AsyncClient(
            base_url=self._worker_url, timeout=timeout, transport=transport
        )

    monkeypatch.setattr(r.BaseRemoteMixin, "_build_client", _build)

    provider = r.RemoteTextProvider(_make_remote_config("text"))
    await provider.load("/tmp")
    assert provider.is_loaded()


@pytest.mark.asyncio
async def test_e2e_async_load_failure_propagates(monkeypatch):
    """FakeWorker reports failed → gateway raises RuntimeError carrying error."""
    from app.providers import remote as r

    monkeypatch.setattr(r, "_LOAD_POLL_BACKOFF", [0.0])

    app = _make_polling_worker([
        {"status": "failed", "error": "OOM at layer 42"},
    ])
    transport = ASGITransport(app=app)

    def _build(self, timeout):
        return httpx.AsyncClient(
            base_url=self._worker_url, timeout=timeout, transport=transport
        )

    monkeypatch.setattr(r.BaseRemoteMixin, "_build_client", _build)

    provider = r.RemoteTextProvider(_make_remote_config("text"))
    with pytest.raises(RuntimeError, match="OOM at layer 42"):
        await provider.load("/tmp")
    assert not provider.is_loaded()


@pytest.mark.asyncio
async def test_e2e_load_total_timeout(monkeypatch):
    """FakeWorker stays in loading forever — gateway raises after _LOAD_TIMEOUT."""
    from app.providers import remote as r

    monkeypatch.setattr(r, "_LOAD_POLL_BACKOFF", [0.05])
    monkeypatch.setattr(r, "_LOAD_TIMEOUT", 0.2)

    app = _make_polling_worker([{"status": "loading"}] * 100)
    transport = ASGITransport(app=app)

    def _build(self, timeout):
        return httpx.AsyncClient(
            base_url=self._worker_url, timeout=timeout, transport=transport
        )

    monkeypatch.setattr(r.BaseRemoteMixin, "_build_client", _build)

    provider = r.RemoteTextProvider(_make_remote_config("text"))
    with pytest.raises(RuntimeError, match="did not become ready"):
        await provider.load("/tmp")
    assert not provider.is_loaded()


@pytest.mark.asyncio
async def test_e2e_already_ready_returns_200_fast_path(monkeypatch):
    """Worker reports model already loaded → /load returns 200 → gateway skips polling."""
    from app.providers import remote as r

    fast = FastAPI()

    @fast.get("/health")
    async def _h():
        return {"status": "ok"}

    @fast.post("/load")
    async def _l():
        return {"status": "ok"}

    transport = ASGITransport(app=fast)

    def _build(self, timeout):
        return httpx.AsyncClient(
            base_url=self._worker_url, timeout=timeout, transport=transport
        )

    monkeypatch.setattr(r.BaseRemoteMixin, "_build_client", _build)

    provider = r.RemoteTextProvider(_make_remote_config("text"))
    await provider.load("/tmp")
    assert provider.is_loaded()


@pytest.mark.asyncio
async def test_e2e_load_cancellation_during_polling(monkeypatch):
    """Outer asyncio.wait_for cancels polling — gateway closes httpx client cleanly."""
    import asyncio as _asyncio

    from app.providers import remote as r

    monkeypatch.setattr(r, "_LOAD_POLL_BACKOFF", [0.5])

    app = _make_polling_worker([{"status": "loading"}] * 100)
    transport = ASGITransport(app=app)

    def _build(self, timeout):
        return httpx.AsyncClient(
            base_url=self._worker_url, timeout=timeout, transport=transport
        )

    monkeypatch.setattr(r.BaseRemoteMixin, "_build_client", _build)

    provider = r.RemoteTextProvider(_make_remote_config("text"))
    with pytest.raises(_asyncio.TimeoutError):
        await _asyncio.wait_for(provider.load("/tmp"), timeout=0.2)

    assert not provider.is_loaded()
    # httpx client must be closed → second load() rebuilds it without error.
    assert provider._client is None


