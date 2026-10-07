from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_cache_stats(client):
    resp = await client.get("/cache/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "global" in data


@pytest.mark.asyncio
async def test_clear_cache(client):
    # Generate something to populate cache
    await client.post(
        "/v1/audio/speech",
        json={"model": "test-tts", "input": "Cached entry"},
    )

    resp = await client.delete("/cache")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] >= 0


@pytest.mark.asyncio
async def test_clear_model_cache(client):
    await client.post(
        "/v1/audio/speech",
        json={"model": "test-tts", "input": "Model cache entry"},
    )

    resp = await client.delete("/cache/test-tts")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_delete_nonexistent_entry(client):
    resp = await client.delete("/cache/entry/nonexistent-key")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_an_image_cache_hit_does_not_load_the_evicted_model(client, services):
    payload = {"model": "test-image", "prompt": "A blue square", "seed": 42, "size": "256x256"}
    first = await client.post("/v1/images/generations", json=payload)
    assert first.headers["x-infergate-cache"] == "MISS"
    await services["manager"].unload_model("test-image")

    second = await client.post("/v1/images/generations", json=payload)

    assert second.headers["x-infergate-cache"] == "HIT"
    assert not services["manager"].get("test-image").is_loaded()


@pytest.mark.asyncio
async def test_a_speech_cache_hit_does_not_load_the_evicted_model(client, services):
    payload = {"model": "test-tts", "input": "Hello again"}
    first = await client.post("/v1/audio/speech", json=payload)
    assert first.headers["x-infergate-cache"] == "MISS"
    await services["manager"].unload_model("test-tts")

    second = await client.post("/v1/audio/speech", json=payload)

    assert second.headers["x-infergate-cache"] == "HIT"
    assert not services["manager"].get("test-tts").is_loaded()
