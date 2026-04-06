from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_image_generation(client):
    resp = await client.post(
        "/v1/images/generations",
        json={
            "model": "test-image",
            "prompt": "A red circle",
            "n": 1,
            "size": "512x512",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) == 1
    assert data["data"][0]["b64_json"] is not None


@pytest.mark.asyncio
async def test_image_cache_with_seed(client):
    payload = {
        "model": "test-image",
        "prompt": "A blue square",
        "seed": 42,
        "size": "256x256",
    }
    # First request — MISS
    resp1 = await client.post("/v1/images/generations", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers["x-infergate-cache"] == "MISS"

    # Second request — HIT
    resp2 = await client.post("/v1/images/generations", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers["x-infergate-cache"] == "HIT"


@pytest.mark.asyncio
async def test_image_no_cache_without_seed(client):
    resp = await client.post(
        "/v1/images/generations",
        json={"model": "test-image", "prompt": "Test"},
    )
    assert resp.status_code == 200
    # seed_only strategy — no seed means no cache
    assert resp.headers["x-infergate-cache"] == "DISABLED"
