"""Per-request queue priority from the X-InferGate-Priority header."""
from __future__ import annotations

import asyncio

import pytest

from app.config import Priority
from app.services.gpu_scheduler import GpuScheduler


@pytest.fixture
def submitted(services, monkeypatch):
    seen: list = []
    scheduler = services["scheduler"]
    original = scheduler.submit

    async def spy(model_id, priority, coro, *args):
        seen.append(priority)
        return await original(model_id, priority, coro, *args)

    monkeypatch.setattr(scheduler, "submit", spy)
    return seen


@pytest.mark.asyncio
async def test_higher_priority_waiter_runs_first():
    scheduler = GpuScheduler(max_queue_size=5)
    scheduler.register_model("test", 1)
    gate = asyncio.Event()
    order: list[str] = []

    async def job(name: str):
        if name == "first":
            await gate.wait()
        order.append(name)

    first = asyncio.create_task(scheduler.submit("test", "medium", job("first"), timeout=5.0))
    await asyncio.sleep(0.01)
    low = asyncio.create_task(scheduler.submit("test", "low", job("low"), timeout=5.0))
    await asyncio.sleep(0.01)
    high = asyncio.create_task(scheduler.submit("test", "high", job("high"), timeout=5.0))
    await asyncio.sleep(0.01)
    gate.set()
    await asyncio.gather(first, low, high)
    assert order == ["first", "high", "low"]


@pytest.mark.asyncio
async def test_header_sets_the_priority(client, submitted):
    resp = await client.post(
        "/v1/images/generations",
        json={"model": "test-image", "prompt": "a cat"},
        headers={"X-InferGate-Priority": "high"},
    )
    assert resp.status_code == 200
    assert submitted == [Priority.HIGH]


@pytest.mark.asyncio
async def test_without_header_the_model_priority_applies(client, submitted):
    resp = await client.post("/v1/audio/speech", json={"model": "test-tts", "input": "hi"})
    assert resp.status_code == 200
    assert submitted == [Priority.MEDIUM]


@pytest.mark.asyncio
async def test_unknown_priority_is_rejected(client, submitted):
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "test-embed-text", "input": "hi"},
        headers={"X-InferGate-Priority": "urgent"},
    )
    assert resp.status_code == 422
    assert submitted == []
