from __future__ import annotations

import pytest

from app.config import ModelCacheConfig, ModelConfig, ModelMetadata, ModelQueueConfig
from app.services.provider_manager import ModelNotFoundError


@pytest.mark.asyncio
async def test_list_models(services):
    manager = services["manager"]
    models = manager.list_models()
    assert len(models) == 5
    ids = {m["id"] for m in models}
    assert ids == {"test-image", "test-text", "test-tts", "test-stt", "test-upscale"}


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
    assert len(data["data"]) == 5


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
async def test_byte_budget_lru_evicts_on_oversubscription(services):
    """Loading a model that would push the sum of declared vram_mb past
    the budget evicts the LRU until it fits — even if the count cap is
    nowhere near its limit."""
    manager = services["manager"]
    # Each fixture model declares 1000 MB. Budget = 2500 MB with 0 headroom
    # → 2 models fit (2 * 1000 = 2000 ≤ 2500), a third triggers eviction
    # even though max_loaded (=3) hasn't been reached.
    manager._max_vram_budget_mb = 2500
    manager._vram_headroom_mb = 0

    await manager.ensure_loaded("test-image")
    await manager.ensure_loaded("test-text")
    assert set(manager.loaded_models()) == {"test-image", "test-text"}

    # Third load: 3 * 1000 > 2500 → byte-budget kicks in, evicts LRU
    await manager.ensure_loaded("test-tts")
    loaded = set(manager.loaded_models())
    assert loaded == {"test-text", "test-tts"}
    assert not manager.get("test-image").is_loaded()  # oldest evicted


@pytest.mark.asyncio
async def test_byte_budget_respects_headroom(services):
    """vram_headroom_mb carves a gap out of the top — pinned + active
    usage may not push past (budget - headroom)."""
    manager = services["manager"]
    # Budget 3000, headroom 500 → effective 2500 → 2 models fit
    manager._max_vram_budget_mb = 3000
    manager._vram_headroom_mb = 500

    await manager.ensure_loaded("test-image")
    await manager.ensure_loaded("test-text")
    await manager.ensure_loaded("test-tts")
    loaded = set(manager.loaded_models())
    assert len(loaded) == 2  # same eviction as above, triggered by headroom


@pytest.mark.asyncio
async def test_byte_budget_raises_when_all_pinned(services):
    """Every model pinned + new load doesn't fit → InsufficientResourcesError.
    Gateway turns this into HTTP 503 (instead of letting CUDA OOM take out
    the worker on a real system)."""
    from app.services.provider_manager import InsufficientResourcesError

    manager = services["manager"]
    manager._max_vram_budget_mb = 1500
    manager._vram_headroom_mb = 0
    await manager.ensure_loaded("test-image")
    manager._pinned.add("test-image")  # freeze the only evictable slot

    # Loading test-text (1000 MB) would push total to 2000 > 1500, but
    # the only loaded model is pinned so LRU can't help.
    with pytest.raises(InsufficientResourcesError, match="Cannot fit"):
        await manager.ensure_loaded("test-text")


@pytest.mark.asyncio
async def test_byte_budget_disabled_falls_back_to_count(services):
    """max_vram_budget_mb=0 → old count-based LRU only."""
    manager = services["manager"]
    manager._max_vram_budget_mb = 0  # disabled
    manager._max_loaded = 2  # hard count cap

    await manager.ensure_loaded("test-image")
    await manager.ensure_loaded("test-text")
    await manager.ensure_loaded("test-tts")
    loaded = set(manager.loaded_models())
    # Count-based LRU evicts LRU (test-image) on the 3rd load
    assert len(loaded) == 2
    assert "test-image" not in loaded


@pytest.mark.asyncio
async def test_validate_config_rejects_pinned_over_budget():
    """Pinning so many models that their declared vram sum ≥ budget
    raises ConfigError at startup — operator must fix before serving."""
    from app.services.provider_manager import ConfigError, ProviderManager

    manager = ProviderManager(
        model_dir=".", max_loaded=5,
        pinned=["test-image", "test-text"],
        max_vram_budget_mb=1500,  # two 1000 MB models = 2000 > 1500
        vram_headroom_mb=0,
    )
    img_cfg = _make_test_config("test-image")
    txt_cfg = _make_test_config("test-text")
    manager._registry["test-image"] = _MinimalProvider(img_cfg)
    manager._registry["test-text"] = _MinimalProvider(txt_cfg)

    with pytest.raises(ConfigError, match="max_vram_budget_mb"):
        manager.validate_config()


class _MinimalProvider:
    """Just enough surface for validate_config to read vram_mb without
    needing a full fake provider fixture."""
    def __init__(self, config):
        self.config = config

    @property
    def vram_mb(self) -> int:
        return self.config.model.get("vram_mb", 0)


@pytest.mark.asyncio
async def test_active_request_counter_increments_and_decrements(services):
    """Context manager pumps the counter exactly once."""
    manager = services["manager"]
    assert manager.active_request_count("test-image") == 0
    async with manager.active_request("test-image"):
        assert manager.active_request_count("test-image") == 1
        async with manager.active_request("test-image"):
            assert manager.active_request_count("test-image") == 2
        assert manager.active_request_count("test-image") == 1
    assert manager.active_request_count("test-image") == 0


@pytest.mark.asyncio
async def test_lru_skips_model_with_active_request(services):
    """Core fix: a model with an in-flight request can't be evicted,
    even if it's the LRU candidate."""
    manager = services["manager"]
    manager._max_loaded = 2  # count cap forces eviction on 3rd load

    await manager.ensure_loaded("test-image")
    await manager.ensure_loaded("test-text")
    # Simulate an in-flight request on "test-image" (the LRU).
    async with manager.active_request("test-image"):
        # Loading test-tts would evict LRU non-active = test-text
        # (skipping test-image because it's busy).
        await manager.ensure_loaded("test-tts")
        assert manager.get("test-image").is_loaded()  # busy — NOT evicted
        assert not manager.get("test-text").is_loaded()  # LRU non-busy → evicted
        assert manager.get("test-tts").is_loaded()


@pytest.mark.asyncio
async def test_lru_falls_back_to_pinned_when_all_busy(services):
    """All non-pinned loaded models are active → no victim, warning path.
    Matches the all-pinned branch — returns None, caller decides."""
    manager = services["manager"]
    await manager.ensure_loaded("test-image")
    await manager.ensure_loaded("test-text")

    async with manager.active_request("test-image"), manager.active_request("test-text"):
        victim = manager._find_lru_victim()
        assert victim is None


@pytest.mark.asyncio
async def test_active_counter_cleans_up_on_exception(services):
    """Exception inside the context decrements counter correctly."""
    manager = services["manager"]
    with pytest.raises(RuntimeError, match="boom"):
        async with manager.active_request("test-image"):
            assert manager.active_request_count("test-image") == 1
            raise RuntimeError("boom")
    assert manager.active_request_count("test-image") == 0


@pytest.mark.asyncio
async def test_provider_get_stats_default(services):
    """BaseProvider.get_stats returns a declared-only snapshot when the
    concrete provider doesn't override (covers every local provider:
    FakeImageProvider, DiffusersImageProvider, etc.)."""
    manager = services["manager"]
    await manager.ensure_loaded("test-image")
    stats = await manager.get("test-image").get_stats()
    assert stats["model"] == "test-image"
    assert stats["loaded"] is True
    assert stats["declared_vram_mb"] == 1000  # from _make_model_config
    assert stats["vram_used_mb"] == 1000  # declared == used when loaded
    # Unloaded → zero
    await manager.unload_model("test-image")
    stats = await manager.get("test-image").get_stats()
    assert stats["vram_used_mb"] == 0


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


# ── Eviction planning / admission control ───────────────────────────

@pytest.mark.asyncio
async def test_plan_eviction_empty_when_already_fits(services):
    """Nothing loaded → empty plan, never None."""
    manager = services["manager"]
    manager._max_vram_budget_mb = 5000
    manager._vram_headroom_mb = 0
    assert manager._plan_eviction(1000) == []


@pytest.mark.asyncio
async def test_plan_eviction_returns_ordered_victims(services):
    """Plan is the LRU walk — oldest loaded appears first."""
    manager = services["manager"]
    manager._max_vram_budget_mb = 2500
    manager._vram_headroom_mb = 0
    await manager.ensure_loaded("test-image")
    await manager.ensure_loaded("test-text")
    # Incoming 1000 MB + currently 2000 MB → need to free 500 MB → evict one.
    plan = manager._plan_eviction(1000)
    assert plan == ["test-image"]


@pytest.mark.asyncio
async def test_plan_eviction_accumulates_until_fits(services):
    """Larger incoming forces multi-step eviction."""
    manager = services["manager"]
    manager._max_vram_budget_mb = 2500
    manager._vram_headroom_mb = 0
    await manager.ensure_loaded("test-image")
    await manager.ensure_loaded("test-text")
    # Incoming 2500 MB → need to free both current 1000 MB slots to fit.
    plan = manager._plan_eviction(2500)
    assert plan == ["test-image", "test-text"]


@pytest.mark.asyncio
async def test_plan_eviction_infeasible_when_all_pinned(services):
    """Every loaded model pinned and still wouldn't fit → None."""
    manager = services["manager"]
    manager._max_vram_budget_mb = 1500
    manager._vram_headroom_mb = 0
    await manager.ensure_loaded("test-image")
    manager._pinned.add("test-image")
    assert manager._plan_eviction(1000) is None


@pytest.mark.asyncio
async def test_plan_eviction_infeasible_when_all_active(services):
    """In-flight models are excluded from plan; no other candidates → None."""
    manager = services["manager"]
    manager._max_vram_budget_mb = 1500
    manager._vram_headroom_mb = 0
    await manager.ensure_loaded("test-image")
    async with manager.active_request("test-image"):
        assert manager._plan_eviction(1000) is None


@pytest.mark.asyncio
async def test_plan_respects_category_reservation_until_forced(services):
    """Plan honours soft reservation in pass 1; fall-through pass 2
    when otherwise infeasible."""
    manager = services["manager"]
    manager._category_reservations = {"text": 1}
    manager._max_vram_budget_mb = 2500
    manager._vram_headroom_mb = 0
    await manager.ensure_loaded("test-text")
    await manager.ensure_loaded("test-image")
    # 1000 incoming → one slot is enough. Pass 1 finds test-image
    # (reservation on text skips it), plan = [test-image].
    assert manager._plan_eviction(1000) == ["test-image"]
    # 2500 incoming → both slots must go. Reservation bends in pass 2
    # when no alternative remains. Plan keeps LRU order.
    assert manager._plan_eviction(2500) == ["test-image", "test-text"]


# ── Category reservations (QoS) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_category_reservation_protects_last_member(services):
    """With text reserved at 1, chat traffic can't evict the only text
    model when an image-category peer is a valid alternative."""
    manager = services["manager"]
    manager._category_reservations = {"text": 1}
    manager._max_loaded = 2

    await manager.ensure_loaded("test-text")   # oldest — normally LRU target
    await manager.ensure_loaded("test-image")  # would be the next victim

    await manager.ensure_loaded("test-tts")    # forces eviction
    # Reservation held: text survives, image (newer but unreserved) evicted.
    assert manager.get("test-text").is_loaded()
    assert not manager.get("test-image").is_loaded()
    assert manager.get("test-tts").is_loaded()


@pytest.mark.asyncio
async def test_category_reservation_violated_when_no_alternative(services):
    """Reservation is soft: if every non-reserved model is pinned/active,
    fall through and evict the reserved category rather than 503."""
    manager = services["manager"]
    manager._category_reservations = {"text": 1}
    manager._max_loaded = 2

    await manager.ensure_loaded("test-text")
    await manager.ensure_loaded("test-image")
    manager._pinned.add("test-image")  # image cannot be touched

    await manager.ensure_loaded("test-tts")
    # No alternative → text evicted despite reservation.
    assert not manager.get("test-text").is_loaded()
    assert manager.get("test-image").is_loaded()
    assert manager.get("test-tts").is_loaded()


@pytest.mark.asyncio
async def test_category_reservation_allows_eviction_above_floor(services):
    """Reservation of 1 does not protect the second+ member — only the
    last one. Two text models loaded → evicting the older one still
    leaves reservation satisfied."""
    manager = services["manager"]
    manager._category_reservations = {"image": 1}
    manager._max_loaded = 2

    # Two models in the same category: FakeImageProvider works fine
    # as a stand-in since categories are derived from config.
    await manager.ensure_loaded("test-image")
    # Simulate a second image-category member by swapping a fixture in.
    from tests.conftest import FakeImageProvider
    second_cfg = manager.get("test-image").config.model_copy(update={"id": "test-image-2"})
    manager._registry["test-image-2"] = FakeImageProvider(second_cfg)
    await manager.ensure_loaded("test-image-2")

    # Loading test-text would force eviction of LRU = test-image.
    # Reservation of 1 image still satisfied (test-image-2 remains).
    await manager.ensure_loaded("test-text")
    assert not manager.get("test-image").is_loaded()
    assert manager.get("test-image-2").is_loaded()
    assert manager.get("test-text").is_loaded()


@pytest.mark.asyncio
async def test_category_reservation_empty_dict_is_noop(services):
    """Empty reservations {} must not change vanilla LRU behaviour."""
    manager = services["manager"]
    manager._category_reservations = {}
    manager._max_loaded = 2

    await manager.ensure_loaded("test-image")
    await manager.ensure_loaded("test-text")
    await manager.ensure_loaded("test-tts")
    # Plain LRU → image (oldest) evicted.
    assert not manager.get("test-image").is_loaded()
    assert manager.get("test-text").is_loaded()
    assert manager.get("test-tts").is_loaded()


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
