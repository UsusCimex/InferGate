"""Tests for remote provider + provider_manager integration."""
from __future__ import annotations

import httpx
import pytest

from app.config import ModelCacheConfig, ModelConfig, ModelMetadata, ModelQueueConfig
from app.services.provider_manager import ProviderManager


def _make_remote_config(category: str) -> ModelConfig:
    return ModelConfig(
        id=f"remote-{category}",
        display_name=f"Remote {category}",
        category=category,
        provider_class="Unused",
        enabled=True,
        worker_url="http://fake-worker:8001",
        model={"hub_id": "test/test", "vram_mb": 0},
        cache=ModelCacheConfig(),
        queue=ModelQueueConfig(),
        metadata=ModelMetadata(),
    )


def test_discover_creates_remote_provider():
    """ProviderManager should create RemoteProvider when worker_url is set."""
    from app.providers.remote import RemoteTextProvider

    manager = ProviderManager(model_dir=".", max_loaded=2)
    config = _make_remote_config("text")
    manager.discover_models([config])

    provider = manager.get("remote-text")
    assert isinstance(provider, RemoteTextProvider)


def test_discover_creates_remote_image_provider():
    from app.providers.remote import RemoteImageProvider

    manager = ProviderManager(model_dir=".", max_loaded=2)
    config = _make_remote_config("image")
    manager.discover_models([config])

    provider = manager.get("remote-image")
    assert isinstance(provider, RemoteImageProvider)


def test_discover_creates_remote_tts_provider():
    from app.providers.remote import RemoteTtsProvider

    manager = ProviderManager(model_dir=".", max_loaded=2)
    config = _make_remote_config("tts")
    manager.discover_models([config])

    provider = manager.get("remote-tts")
    assert isinstance(provider, RemoteTtsProvider)


def test_remote_model_not_gpu():
    """Remote models with vram_mb=0 should not count toward GPU slots."""
    manager = ProviderManager(model_dir=".", max_loaded=2)
    config = _make_remote_config("text")
    manager.discover_models([config])

    assert not manager._is_gpu_model("remote-text")


def test_mixed_local_and_remote():
    """Can register both local and remote models in the same manager."""
    from tests.conftest import FakeTextProvider, _make_model_config

    manager = ProviderManager(model_dir=".", max_loaded=2)

    local_config = _make_model_config("local-text", "text", "FakeTextProvider")
    remote_config = _make_remote_config("text")

    # Register local manually (skip provider class resolution)
    manager._registry["local-text"] = FakeTextProvider(local_config)
    manager.discover_models([remote_config])

    assert "local-text" in manager._registry
    assert "remote-text" in manager._registry
    assert len(manager.list_models()) == 2


@pytest.mark.asyncio
async def test_retry_send_succeeds_after_transient_failure(monkeypatch):
    """_retry_send must retry ConnectError and succeed once the worker recovers."""
    from app.providers import remote as r

    monkeypatch.setattr(r, "_RETRY_BASE_BACKOFF", 0.0)

    attempts = {"n": 0}

    async def factory():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("connection refused")
        # Pretend we got a real response.
        return httpx.Response(200, content=b"ok")

    resp = await r._retry_send("POST", "/generate", factory)
    assert resp.status_code == 200
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_retry_send_does_not_retry_non_transient(monkeypatch):
    """_retry_send must NOT retry ReadError (request likely already executed)."""
    from app.providers import remote as r

    monkeypatch.setattr(r, "_RETRY_BASE_BACKOFF", 0.0)

    attempts = {"n": 0}

    async def factory():
        attempts["n"] += 1
        raise httpx.ReadError("mid-stream failure")

    with pytest.raises(httpx.ReadError):
        await r._retry_send("POST", "/generate", factory)
    assert attempts["n"] == 1


@pytest.mark.asyncio
async def test_retry_send_exhausts_attempts(monkeypatch):
    """When the worker stays unreachable, _retry_send raises after N attempts."""
    from app.providers import remote as r

    monkeypatch.setattr(r, "_RETRY_BASE_BACKOFF", 0.0)
    monkeypatch.setattr(r, "_RETRY_ATTEMPTS", 3)

    attempts = {"n": 0}

    async def factory():
        attempts["n"] += 1
        raise httpx.ConnectError("down")

    with pytest.raises(httpx.ConnectError):
        await r._retry_send("POST", "/generate", factory)
    assert attempts["n"] == 3


def test_remote_request_id_headers_returns_dict_when_set():
    """_request_id_headers must include X-Request-ID when ContextVar is set."""
    from app.monitoring import set_request_id
    from app.providers.remote import _request_id_headers

    set_request_id("abc123")
    try:
        assert _request_id_headers() == {"X-Request-ID": "abc123"}
    finally:
        set_request_id(None)


def test_remote_request_id_headers_empty_when_unset():
    """_request_id_headers must be empty when no request_id is set."""
    from app.monitoring import set_request_id
    from app.providers.remote import _request_id_headers

    set_request_id(None)
    assert _request_id_headers() == {}
