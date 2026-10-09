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


def test_resolve_worker_url_env_first(monkeypatch):
    """env WORKER_URL_<ID> wins over template."""
    from app.services.provider_manager import resolve_worker_url

    monkeypatch.setenv("WORKER_URL_QWEN3_5_4B", "http://from-env:8001")
    url = resolve_worker_url(
        "qwen3.5-4b", template="http://discovery/{id}:8000"
    )
    assert url == "http://from-env:8001"


def test_resolve_worker_url_template_fallback(monkeypatch):
    """When env is missing, the template is used."""
    from app.services.provider_manager import resolve_worker_url

    monkeypatch.delenv("WORKER_URL_QWEN3_5_4B", raising=False)
    url = resolve_worker_url(
        "qwen3.5-4b", template="http://worker-{id}.workers:8000"
    )
    assert url == "http://worker-qwen3.5-4b.workers:8000"


def test_resolve_worker_url_returns_none_without_template_or_env(monkeypatch):
    from app.services.provider_manager import resolve_worker_url

    monkeypatch.delenv("WORKER_URL_LOCAL_ONLY", raising=False)
    assert resolve_worker_url("local-only", template=None) is None


def test_resolve_worker_url_handles_special_chars_in_id(monkeypatch):
    """env key normalisation: dots and dashes collapse to underscores; uppercased."""
    from app.services.provider_manager import resolve_worker_url

    monkeypatch.setenv("WORKER_URL_FLUX1_SCHNELL_FP8", "http://flux:8001")
    url = resolve_worker_url("flux1-schnell.fp8", template=None)
    assert url == "http://flux:8001"


def test_resolve_worker_url_bad_template_returns_none(monkeypatch, caplog):
    """A template that references a non-existent placeholder logs a warning, returns None."""
    import logging

    from app.services.provider_manager import resolve_worker_url

    monkeypatch.delenv("WORKER_URL_X", raising=False)
    with caplog.at_level(logging.WARNING):
        url = resolve_worker_url("x", template="http://worker-{nonexistent}:8000")
    assert url is None
    assert any("could not be formatted" in r.message for r in caplog.records)


def test_provider_manager_uses_template_for_remote_discovery():
    """ProviderManager.discover_models picks up template-derived worker URLs."""
    from app.services.provider_manager import ProviderManager

    manager = ProviderManager(
        model_dir=".", max_loaded=2,
        worker_url_template="http://worker-{id}:8000",
    )
    cfg = ModelConfig(
        id="text-template",
        display_name="t",
        category="text",
        provider_class="Unused",
        enabled=True,
        worker_url=None,  # neither explicit nor env
        model={"hub_id": "test/test", "vram_mb": 0},
        cache=ModelCacheConfig(),
        queue=ModelQueueConfig(),
        metadata=ModelMetadata(),
    )
    manager.discover_models([cfg])

    provider = manager.get("text-template")
    assert provider.config.worker_url == "http://worker-text-template:8000"


@pytest.mark.asyncio
async def test_reload_of_an_unchanged_yaml_keeps_a_template_worker(monkeypatch):
    """The YAML has no worker_url; the template fills one in, which must not count as a change."""
    from app.services.provider_manager import ProviderManager

    manager = ProviderManager(
        model_dir=".", max_loaded=2, worker_url_template="http://worker-{id}:8000",
    )

    def yaml_config() -> ModelConfig:
        return ModelConfig(
            id="text-template", display_name="t", category="text", provider_class="Unused",
            model={"hub_id": "test/test", "vram_mb": 0},
        )

    manager.discover_models([yaml_config()])
    before = manager.get("text-template")
    assert await manager.reload_model(yaml_config()) is False
    assert manager.get("text-template") is before


_KEEPALIVE_RESPONSE = [b"HTTP/1.1 200 OK\r\n", b"Content-Length: 2\r\n", b"\r\n", b"ok"]


def _pooled_client(max_connections: int = 10) -> httpx.AsyncClient:
    """httpx client over a real httpcore pool whose sockets are replaced by canned responses."""
    import httpcore

    transport = httpx.AsyncHTTPTransport()
    transport._pool = httpcore.AsyncConnectionPool(
        max_connections=max_connections,
        network_backend=httpcore.AsyncMockBackend(_KEEPALIVE_RESPONSE * 2),
    )
    return httpx.AsyncClient(transport=transport, base_url="http://worker")


@pytest.mark.asyncio
async def test_pool_stats_tracks_active_and_idle_connections():
    from app.providers.remote import _pool_stats

    async with _pooled_client() as client:
        assert _pool_stats(client) == {"active": 0, "idle": 0, "waiting": 0}
        async with client.stream("GET", "/health") as resp:
            assert _pool_stats(client) == {"active": 1, "idle": 0, "waiting": 0}
            await resp.aread()
        assert _pool_stats(client) == {"active": 0, "idle": 1, "waiting": 0}


@pytest.mark.asyncio
async def test_pool_stats_counts_requests_waiting_for_a_connection():
    import asyncio

    from app.providers.remote import _pool_stats

    async with _pooled_client(max_connections=1) as client:
        async with client.stream("GET", "/first") as first:
            second = asyncio.create_task(client.get("/second"))
            for _ in range(5):
                await asyncio.sleep(0)
            assert _pool_stats(client) == {"active": 1, "idle": 0, "waiting": 1}
            await first.aread()
        assert (await second).text == "ok"


def test_pool_stats_is_none_without_a_connection_pool():
    from app.providers.remote import _pool_stats

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    assert _pool_stats(client) is None


def test_pool_stats_is_none_when_httpcore_internals_change():
    from types import SimpleNamespace

    from app.providers.remote import _pool_stats

    renamed = SimpleNamespace(is_available=lambda: True)
    pool = SimpleNamespace(connections=[renamed], _requests=[])
    assert _pool_stats(SimpleNamespace(_transport=SimpleNamespace(_pool=pool))) is None
