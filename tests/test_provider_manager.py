from __future__ import annotations

import pytest

from app.config import ModelCacheConfig, ModelConfig, ModelMetadata, ModelQueueConfig
from app.services.provider_manager import ModelNotFoundError


@pytest.mark.asyncio
async def test_list_models(services):
    manager = services["manager"]
    models = manager.list_models()
    assert len(models) == 4
    ids = {m["id"] for m in models}
    assert ids == {"test-image", "test-text", "test-tts", "test-stt"}


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
    assert len(data["data"]) == 4


# ── Hot-reload coverage ──────────────────────────────────────────────

def _make_test_config(
    model_id: str = "test-image",
    display_name: str = "test-image",
    max_concurrent: int = 2,
    enabled: bool = True,
    provider_class: str = "FakeImageProvider",
) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        display_name=display_name,
        category="image",
        provider_class=provider_class,
        enabled=enabled,
        model={"hub_id": "test/test", "vram_mb": 1000},
        cache=ModelCacheConfig(enabled=False, strategy="never", max_size_mb=100),
        queue=ModelQueueConfig(priority="medium", timeout_seconds=30, max_concurrent=max_concurrent),
        metadata=ModelMetadata(),
    )


@pytest.mark.asyncio
async def test_reload_noop_for_identical_config(services):
    """Saving a YAML without content changes must not swap the provider."""
    manager = services["manager"]
    before = manager.get("test-image")
    same_config = before.config  # byte-equal to the live config
    changed = await manager.reload_model(same_config)
    assert changed is False
    assert manager.get("test-image") is before  # same instance, not replaced


@pytest.mark.asyncio
async def test_reload_swaps_provider_on_change(services):
    """Changing any field must replace the provider while preserving id."""
    manager = services["manager"]
    before = manager.get("test-image")
    new_config = _make_test_config(display_name="renamed")
    changed = await manager.reload_model(new_config)
    assert changed is True
    after = manager.get("test-image")
    assert after is not before  # fresh provider instance
    assert after.config.display_name == "renamed"


@pytest.mark.asyncio
async def test_reload_preserves_loaded_state(services):
    """A provider that was loaded stays loaded after reload — operators
    don't want `systemctl restart` just to edit a display_name."""
    manager = services["manager"]
    await manager.ensure_loaded("test-image")
    assert manager.get("test-image").is_loaded()

    new_config = _make_test_config(display_name="renamed")
    await manager.reload_model(new_config)

    reloaded = manager.get("test-image")
    assert reloaded.is_loaded()
    assert "test-image" in manager.loaded_models()


@pytest.mark.asyncio
async def test_reload_disables_and_unregisters(services):
    """enabled: false in YAML → model disappears from the registry and
    from /v1/models after the next reload."""
    manager = services["manager"]
    await manager.ensure_loaded("test-image")

    disabled_config = _make_test_config(enabled=False)
    changed = await manager.reload_model(disabled_config)
    assert changed is True

    with pytest.raises(ModelNotFoundError):
        manager.get("test-image")
    assert "test-image" not in manager.loaded_models()


@pytest.mark.asyncio
async def test_reload_registers_new_model(services):
    """A YAML file for a previously-unknown model_id registers it fresh."""
    manager = services["manager"]
    new_config = _make_test_config(model_id="test-new-image", display_name="new")
    changed = await manager.reload_model(new_config)
    assert changed is True
    assert manager.get("test-new-image").config.display_name == "new"


@pytest.mark.asyncio
async def test_reload_disabled_new_model_is_noop(services):
    """A YAML with enabled: false for a model we don't have must not
    register a disabled placeholder."""
    manager = services["manager"]
    config = _make_test_config(model_id="test-disabled", enabled=False)
    changed = await manager.reload_model(config)
    assert changed is False
    with pytest.raises(ModelNotFoundError):
        manager.get("test-disabled")


@pytest.mark.asyncio
async def test_reload_bad_provider_class_keeps_old(services):
    """Typo in provider_class on reload must not lose the existing
    provider — the operator should be able to fix the YAML and try again."""
    manager = services["manager"]
    before = manager.get("test-image")
    bad_config = _make_test_config(provider_class="DoesNotExist")
    changed = await manager.reload_model(bad_config)
    assert changed is False
    # Still accessible under the old provider
    assert manager.get("test-image") is before


@pytest.mark.asyncio
async def test_scheduler_update_concurrency(services):
    """Hot-reload of queue.max_concurrent updates the live slot in place."""
    scheduler = services["scheduler"]
    assert scheduler._queues["test-image"].max_concurrent == 2
    scheduler.update_concurrency("test-image", 5)
    assert scheduler._queues["test-image"].max_concurrent == 5


@pytest.mark.asyncio
async def test_scheduler_update_concurrency_registers_unknown(services):
    """update_concurrency on an unknown model_id transparently registers
    the queue — avoids a race when reload_model creates the model and
    the callback calls update_concurrency before register_model has run."""
    scheduler = services["scheduler"]
    scheduler.update_concurrency("brand-new", 3)
    assert scheduler._queues["brand-new"].max_concurrent == 3


# ── Remote-provider reload path ──────────────────────────────────────

class _FakeRemoteProvider:
    """Records reload() calls without hitting HTTP."""
    def __init__(self, config):
        self.config = config
        self._loaded = True
        self.reload_calls: list = []

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vram_mb(self) -> int:
        return 0

    @property
    def model_id(self) -> str:
        return self.config.id

    async def reload(self, new_config) -> str:
        self.reload_calls.append(new_config)
        self.config = new_config
        return "metadata"


@pytest.mark.asyncio
async def test_reload_remote_calls_worker_endpoint(services):
    """Connected remote + unchanged worker_url → reload_model routes via existing.reload()."""
    manager = services["manager"]

    remote_cfg = _make_test_config()
    remote_cfg.worker_url = "http://worker-test:8001"
    fake = _FakeRemoteProvider(remote_cfg)
    manager._registry["test-image"] = fake

    new_cfg = _make_test_config(display_name="renamed")
    new_cfg.worker_url = "http://worker-test:8001"
    changed = await manager.reload_model(new_cfg)
    assert changed is True

    # Same provider instance — not replaced with a fresh RemoteProvider
    assert manager.get("test-image") is fake
    # Worker /reload was called exactly once with the new config
    assert len(fake.reload_calls) == 1
    assert fake.reload_calls[0].display_name == "renamed"
    assert fake.config.display_name == "renamed"


@pytest.mark.asyncio
async def test_reload_remote_falls_back_on_worker_error(services):
    """Worker /reload failure → recreate gateway-side provider so model stays registered."""
    manager = services["manager"]

    class _FailingRemote(_FakeRemoteProvider):
        async def reload(self, new_config):
            raise RuntimeError("worker unreachable")

    remote_cfg = _make_test_config()
    remote_cfg.worker_url = "http://worker-test:8001"
    failing = _FailingRemote(remote_cfg)
    manager._registry["test-image"] = failing

    new_cfg = _make_test_config(display_name="renamed")
    new_cfg.worker_url = "http://worker-test:8001"
    changed = await manager.reload_model(new_cfg)

    # The remote path failed, but the fallback rebuilt the remote
    # provider so the model is still registered. `changed` is True:
    # something WAS updated (the provider instance is fresh).
    assert changed is True
    assert "test-image" in manager._registry
