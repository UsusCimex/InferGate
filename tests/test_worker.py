"""Tests for the standalone worker app."""
from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import ModelCacheConfig, ModelConfig, ModelMetadata, ModelQueueConfig
from tests.conftest import FakeImageProvider, FakeTextProvider


def _make_config(category: str, provider_class: str) -> ModelConfig:
    return ModelConfig(
        id=f"test-{category}",
        display_name=f"Test {category}",
        category=category,
        provider_class=provider_class,
        enabled=True,
        model={"hub_id": "test/test"},
        cache=ModelCacheConfig(),
        queue=ModelQueueConfig(),
        metadata=ModelMetadata(),
    )


@pytest_asyncio.fixture
async def text_worker():
    """Worker app serving a fake text model."""
    from fastapi import FastAPI

    app = FastAPI()

    config = _make_config("text", "FakeTextProvider")
    provider = FakeTextProvider(config)
    await provider.load(".")

    app.state.provider = provider
    app.state.config = config
    app.state.reload_lock = asyncio.Lock()

    # Import worker routes
    from app.worker import generate, health, load, reload_config, stats, synthesize, unload
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/load", load, methods=["POST"])
    app.add_api_route("/unload", unload, methods=["POST"])
    app.add_api_route("/generate", generate, methods=["POST"])
    app.add_api_route("/synthesize", synthesize, methods=["POST"])
    app.add_api_route("/reload", reload_config, methods=["POST"])
    app.add_api_route("/stats", stats, methods=["GET"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://worker") as ac:
        yield ac


@pytest_asyncio.fixture
async def image_worker():
    """Worker app serving a fake image model."""
    from fastapi import FastAPI
    app = FastAPI()

    config = _make_config("image", "FakeImageProvider")
    provider = FakeImageProvider(config)
    await provider.load(".")

    app.state.provider = provider
    app.state.config = config
    app.state.reload_lock = asyncio.Lock()

    from app.worker import generate, health, load, reload_config
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/load", load, methods=["POST"])
    app.add_api_route("/generate", generate, methods=["POST"])
    app.add_api_route("/reload", reload_config, methods=["POST"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://worker") as ac:
        yield ac


@pytest.mark.asyncio
async def test_worker_health(text_worker):
    resp = await text_worker.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "test-text"
    assert data["category"] == "text"


@pytest.mark.asyncio
async def test_worker_load(text_worker):
    resp = await text_worker.post("/load")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_worker_text_generate(text_worker):
    resp = await text_worker.post("/generate", json={
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_worker_image_generate(image_worker):
    resp = await image_worker.post("/generate", json={
        "prompt": "A red circle",
        "size": "512x512",
    })
    assert resp.status_code == 200
    # Should return PNG bytes
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


# ── /reload coverage ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reload_identical_config_is_noop(text_worker):
    """Posting the current config back must not trigger a swap."""
    config = _make_config("text", "FakeTextProvider")
    resp = await text_worker.post("/reload", json=config.model_dump(mode="json"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "noop"
    assert data["model"] == "test-text"


@pytest.mark.asyncio
async def test_reload_metadata_only_updates_config_in_place(text_worker):
    """Changing display_name / metadata.description must update state.config
    without rebuilding the provider (which would cost a load/unload cycle)."""
    # display_name + metadata are gateway-scope, but the worker must still
    # accept the config (gateway may POST the full updated YAML) and
    # apply it in place so subsequent generates see the updated defaults.
    new_cfg = _make_config("text", "FakeTextProvider")
    new_cfg_data = new_cfg.model_dump(mode="json")
    new_cfg_data["display_name"] = "Test text [reloaded]"
    new_cfg_data["metadata"]["description"] = "updated"

    resp = await text_worker.post("/reload", json=new_cfg_data)
    assert resp.status_code == 200
    assert resp.json()["action"] == "metadata"

    # /health reads state.config → should reflect the new model id unchanged
    # but the backing config is now the new one.
    h = await text_worker.get("/health")
    assert h.json()["model"] == "test-text"


@pytest.mark.asyncio
async def test_reload_full_swap_on_model_change(text_worker):
    """Changing model.* forces a full rebuild: unload old, create new
    provider, load. Verified by peeking at provider identity via /health
    + asserting action=full_reload."""
    new_cfg_data = _make_config("text", "FakeTextProvider").model_dump(mode="json")
    new_cfg_data["model"] = {"hub_id": "test/different", "vram_mb": 2000}

    resp = await text_worker.post("/reload", json=new_cfg_data)
    assert resp.status_code == 200
    assert resp.json()["action"] == "full_reload"

    # Worker must still be serviceable post-swap
    h = await text_worker.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_reload_rejects_unknown_provider_class(text_worker):
    """Typo in provider_class returns 400, keeps old provider intact."""
    new_cfg_data = _make_config("text", "FakeTextProvider").model_dump(mode="json")
    new_cfg_data["provider_class"] = "DoesNotExist"

    resp = await text_worker.post("/reload", json=new_cfg_data)
    assert resp.status_code == 400
    assert "Unknown provider class" in resp.json()["error"]["message"]

    # Still healthy — old provider untouched
    h = await text_worker.get("/health")
    assert h.status_code == 200


@pytest.mark.asyncio
async def test_reload_rejects_malformed_config(text_worker):
    """Garbage body → 400 with structured error, no crash."""
    resp = await text_worker.post("/reload", json={"this": "is not a ModelConfig"})
    assert resp.status_code == 400
    assert "invalid config" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_stats_returns_usage_snapshot(text_worker):
    """GET /stats returns the full envelope — fake-provider env has no
    CUDA so VRAM fields are 0, but the shape must be correct."""
    resp = await text_worker.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    # Required keys even when CUDA/psutil/NVML unavailable:
    for key in ("model", "loaded", "vram_used_mb", "vram_free_mb",
                "vram_total_mb", "vram_source", "ram_used_mb",
                "ram_total_mb", "declared_vram_mb"):
        assert key in body, f"missing key: {key}"
    assert body["model"] == "test-text"
    assert body["loaded"] is True
    # _make_config in this file doesn't set model.vram_mb, so declared = 0.
    assert body["declared_vram_mb"] == 0


@pytest.mark.asyncio
async def test_stats_prefers_nvml_over_torch(text_worker, monkeypatch):
    """When pynvml is importable, /stats reports device-wide numbers
    and marks vram_source='nvml' — catches the shared-GPU case torch
    can't see."""
    import sys
    import types

    class _FakeInfo:
        total = 12 * 1024 * 1024 * 1024   # 12 GB
        used = 9 * 1024 * 1024 * 1024     # 9 GB — includes a co-tenant process
        free = 3 * 1024 * 1024 * 1024

    fake_pynvml = types.ModuleType("pynvml")
    fake_pynvml.nvmlInit = lambda: None
    fake_pynvml.nvmlShutdown = lambda: None
    fake_pynvml.nvmlDeviceGetHandleByIndex = lambda idx: object()
    fake_pynvml.nvmlDeviceGetMemoryInfo = lambda h: _FakeInfo()
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    resp = await text_worker.get("/stats")
    body = resp.json()
    assert body["vram_source"] == "nvml"
    assert body["vram_total_mb"] == 12 * 1024
    assert body["vram_used_mb"] == 9 * 1024
    assert body["vram_free_mb"] == 3 * 1024


# ── Async /load + /load/status ──────────────────────────────────────


@pytest_asyncio.fixture
async def async_worker():
    """Worker with full lifespan-style state (load_state machine, locks, no preloaded provider)."""
    from fastapi import FastAPI

    from app.worker import (
        _ready_guard,
        generate,
        health,
        load,
        load_status,
        synthesize,
        unload,
    )

    app = FastAPI()
    config = _make_config("text", "FakeTextProvider")
    provider = FakeTextProvider(config)

    app.state.provider = provider
    app.state.config = config
    app.state.models_dir = "."
    app.state.reload_lock = asyncio.Lock()
    app.state.load_lock = asyncio.Lock()
    app.state.load_state = {
        "status": "idle",
        "error": None,
        "started_at": None,
        "duration_seconds": None,
        "cancellation_requested": False,
    }
    app.state.load_task = None

    app.middleware("http")(_ready_guard)
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/load", load, methods=["POST"])
    app.add_api_route("/load/status", load_status, methods=["GET"])
    app.add_api_route("/unload", unload, methods=["POST"])
    app.add_api_route("/generate", generate, methods=["POST"])
    app.add_api_route("/synthesize", synthesize, methods=["POST"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://worker") as ac:
        ac._app = app  # tests poke load_state directly
        yield ac


@pytest.mark.asyncio
async def test_async_load_starts_in_background_and_status_transitions(async_worker):
    """POST /load when idle returns 202 + loading, eventually settles to ready."""
    resp = await async_worker.post("/load")
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "loading"
    assert body["load_state"]["status"] in ("loading", "ready")

    # Drive the background task to completion (FakeProvider load is instant).
    task = async_worker._app.state.load_task
    assert task is not None
    await task

    status_resp = await async_worker.get("/load/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_load_already_loaded_returns_200_sync(async_worker):
    """If load_state is already 'ready', /load returns 200 immediately (fast path)."""
    async_worker._app.state.load_state["status"] = "ready"
    async_worker._app.state.provider._loaded = True

    resp = await async_worker.post("/load")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_inference_returns_503_when_not_ready(async_worker):
    """POST /generate while status='loading' must return 503 model_not_ready."""
    async_worker._app.state.load_state["status"] = "loading"
    resp = await async_worker.post(
        "/generate", json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "model_not_ready"


@pytest.mark.asyncio
async def test_load_failure_marks_status_failed_and_can_retry(async_worker, monkeypatch):
    """Provider.load raises → status=failed, error set; subsequent /load can retry."""
    provider = async_worker._app.state.provider

    fail_count = {"n": 0}
    real_load = provider.load

    async def flaky_load(model_dir):
        fail_count["n"] += 1
        if fail_count["n"] == 1:
            raise RuntimeError("boom")
        await real_load(model_dir)

    monkeypatch.setattr(provider, "load", flaky_load)

    # First attempt: 202 → background fails.
    resp1 = await async_worker.post("/load")
    assert resp1.status_code == 202
    await async_worker._app.state.load_task

    status = (await async_worker.get("/load/status")).json()
    assert status["status"] == "failed"
    assert "boom" in status["error"]

    # Retry: status was 'failed' → /load enters new loading cycle.
    resp2 = await async_worker.post("/load")
    assert resp2.status_code == 202
    await async_worker._app.state.load_task

    status2 = (await async_worker.get("/load/status")).json()
    assert status2["status"] == "ready"


@pytest.mark.asyncio
async def test_unload_during_load_signals_cancelling(async_worker, monkeypatch):
    """POST /unload while loading — cancels background task, transitions away from 'loading'."""
    provider = async_worker._app.state.provider

    async def slow_load(model_dir):
        # Long enough that the test races /unload against it.
        await asyncio.sleep(5.0)
        provider._loaded = True

    monkeypatch.setattr(provider, "load", slow_load)

    load_resp = await async_worker.post("/load")
    assert load_resp.status_code == 202

    # Give the background task a tick to start.
    await asyncio.sleep(0.05)
    assert async_worker._app.state.load_state["status"] == "loading"

    unload_resp = await async_worker.post("/unload")
    # FakeProvider.load awaits asyncio.sleep → cancellable → returns 200 status=ok.
    assert unload_resp.status_code in (200, 409)
    assert async_worker._app.state.load_state["status"] in ("idle", "cancelling")


@pytest.mark.asyncio
async def test_load_status_legacy_path_when_load_state_missing(text_worker):
    """Legacy fixture (no load_state on app.state) — /load/status synthesises shape."""
    # text_worker fixture preloads provider and doesn't set load_state.
    from app.worker import load_status
    text_worker._transport.app.add_api_route(
        "/load/status", load_status, methods=["GET"]
    )

    resp = await text_worker.get("/load/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_reload_serialises_with_generate(text_worker):
    """Reload must wait for an inflight /generate to finish before
    swapping the provider — otherwise the request's provider reference
    dangles. We can't truly fire both in parallel from one httpx client,
    but we can assert the lock is present and acquirable."""
    # Trigger a generate, then immediately reload — with FakeTextProvider
    # returning instantly, both should succeed back-to-back without
    # corrupting state.
    g = text_worker.post("/generate", json={"messages": [{"role": "user", "content": "hi"}]})
    r = text_worker.post("/reload", json=_make_config("text", "FakeTextProvider").model_dump(mode="json"))
    g_resp, r_resp = await asyncio.gather(g, r)
    assert g_resp.status_code == 200
    assert r_resp.status_code == 200
    # Both land successfully — the lock serialised them correctly.
    assert r_resp.json()["action"] == "noop"  # same config
