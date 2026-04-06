from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_tts(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "test-tts", "input": "Hello world"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_tts_cache(client):
    payload = {"model": "test-tts", "input": "Cache test", "voice": "default"}

    resp1 = await client.post("/v1/audio/speech", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers["x-infergate-cache"] == "MISS"

    resp2 = await client.post("/v1/audio/speech", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers["x-infergate-cache"] == "HIT"


@pytest.mark.asyncio
async def test_tts_skip_cache(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "test-tts", "input": "No cache"},
        headers={"X-InferGate-No-Cache": "true"},
    )
    assert resp.status_code == 200
    assert resp.headers["x-infergate-cache"] == "SKIP"
