from __future__ import annotations

import pytest

from app.services.provider_manager import ModelNotFoundError


@pytest.mark.asyncio
async def test_list_models(services):
    manager = services["manager"]
    models = manager.list_models()
    assert len(models) == 3
    ids = {m["id"] for m in models}
    assert ids == {"test-image", "test-text", "test-tts"}


@pytest.mark.asyncio
async def test_ensure_loaded(services):
    manager = services["manager"]
    provider = await manager.ensure_loaded("test-image")
    assert provider.is_loaded()
    assert "test-image" in manager.loaded_models()


@pytest.mark.asyncio
async def test_unload_model(services):
    manager = services["manager"]
    await manager.ensure_loaded("test-image")
    await manager.unload_model("test-image")
    assert not manager.get("test-image").is_loaded()
    assert "test-image" not in manager.loaded_models()


@pytest.mark.asyncio
async def test_lru_eviction(services):
    manager = services["manager"]
    manager._max_loaded = 2

    await manager.ensure_loaded("test-image")
    await manager.ensure_loaded("test-text")
    # This should evict test-image (LRU)
    await manager.ensure_loaded("test-tts")

    assert not manager.get("test-image").is_loaded()
    assert manager.get("test-text").is_loaded()
    assert manager.get("test-tts").is_loaded()


@pytest.mark.asyncio
async def test_model_not_found(services):
    manager = services["manager"]
    with pytest.raises(ModelNotFoundError):
        manager.get("nonexistent")


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_list_models_endpoint(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 3
