"""Tests for remote provider + provider_manager integration."""
from __future__ import annotations

import pytest

from app.config import ModelConfig, ModelCacheConfig, ModelQueueConfig, ModelMetadata
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
