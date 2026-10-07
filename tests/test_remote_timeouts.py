"""Gateway ↔ worker waits: the model's queue timeout governs, a worker timeout is a 504, SSE events stay apart."""
from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient

from app.config import ModelCacheConfig, ModelConfig, ModelMetadata, ModelQueueConfig
from app.providers import remote as remote_module


def _config(category: str, queue_timeout_s: int) -> ModelConfig:
    return ModelConfig(
        id=f"t-{category}", display_name="T", category=category, provider_class="Unused", enabled=True,
        worker_url="http://fake-worker", model={"hub_id": "test/test", "vram_mb": 0},
        cache=ModelCacheConfig(), queue=ModelQueueConfig(timeout_seconds=queue_timeout_s), metadata=ModelMetadata(),
    )


def _worker() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/load")
    async def load():
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(request: Request):
        async def events():
            yield b"data: {\"a\": 1}\n\n"
            yield b"data: {\"b\": 2}\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _use_worker(monkeypatch, app: FastAPI, seen: list[httpx.Timeout]) -> None:
    transport = ASGITransport(app=app)

    def build(self, timeout):
        seen.append(timeout)
        return httpx.AsyncClient(base_url=self._worker_url, timeout=timeout, transport=transport)

    monkeypatch.setattr(remote_module.BaseRemoteMixin, "_build_client", build)


@pytest.mark.asyncio
async def test_a_slow_model_waits_for_the_worker_as_long_as_its_queue_allows(monkeypatch):
    seen: list[httpx.Timeout] = []
    _use_worker(monkeypatch, _worker(), seen)

    await remote_module.RemoteImageProvider(_config("image", 1200)).load("/tmp")
    await remote_module.RemoteImageProvider(_config("image", 30)).load("/tmp")

    assert seen[0].read == 1200
    assert seen[1].read == remote_module._GENERATE_TIMEOUT


@pytest.mark.asyncio
async def test_a_streamed_answer_keeps_the_blank_lines_between_events(monkeypatch):
    _use_worker(monkeypatch, _worker(), [])
    provider = remote_module.RemoteTextProvider(_config("text", 120))
    await provider.load("/tmp")

    streamed = "".join([line async for line in provider.generate_stream([{"role": "user", "content": "hi"}])])

    assert streamed == "data: {\"a\": 1}\n\ndata: {\"b\": 2}\n\ndata: [DONE]\n\n"


@pytest.mark.asyncio
async def test_a_worker_that_does_not_answer_in_time_gives_504():
    from app.main import create_app

    app = create_app()

    @app.get("/slow-worker")
    async def slow_worker():
        raise httpx.ReadTimeout("no answer")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/slow-worker")

    assert resp.status_code == 504
    assert resp.json()["error"]["type"] == "timeout"
