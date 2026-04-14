"""Tests for health and metrics endpoints."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_metrics(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "queue_size" in data
    assert "max_queue_size" in data
    assert "loaded_models" in data
    assert "cache_hit_rate_percent" in data
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_models_list(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 3  # test-image, test-text, test-tts


@pytest.mark.asyncio
async def test_model_load_unload(client):
    # Load
    resp = await client.post("/v1/models/test-text/load")
    assert resp.status_code == 200
    assert resp.json()["loaded"] is True

    # Unload
    resp = await client.post("/v1/models/test-text/unload")
    assert resp.status_code == 200
    assert resp.json()["loaded"] is False


@pytest.mark.asyncio
async def test_model_load_not_found(client):
    resp = await client.post("/v1/models/nonexistent/load")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_model_unload_not_found(client):
    resp = await client.post("/v1/models/nonexistent/unload")
    assert resp.status_code == 404
