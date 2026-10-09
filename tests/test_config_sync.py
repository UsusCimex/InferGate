from __future__ import annotations

import asyncio
import json
import logging

import fakeredis
import pytest

from app.config import ModelConfig
from app.services import config_sync
from app.services.config_sync import ConfigSync

_CHANNEL = "infergate:test-config"


def _config(**overrides) -> ModelConfig:
    fields = {
        "id": "sync-model",
        "display_name": "first",
        "category": "image",
        "provider_class": "FakeImageProvider",
        "model": {"hub_id": "test/test"},
    }
    return ModelConfig(**{**fields, **overrides})


class _Applied:
    def __init__(self) -> None:
        self.configs: list[ModelConfig] = []
        self.event = asyncio.Event()

    async def __call__(self, config: ModelConfig) -> None:
        self.configs.append(config)
        self.event.set()


async def _gateway(server, apply) -> ConfigSync:
    sync = ConfigSync("redis://unused", _CHANNEL, apply, client=fakeredis.FakeAsyncRedis(server=server))
    sync.start()
    await asyncio.wait_for(sync.subscribed.wait(), 2)
    return sync


async def test_a_change_reaches_the_other_gateway_and_not_back():
    server = fakeredis.FakeServer()
    local, remote = _Applied(), _Applied()
    a, b = await _gateway(server, local), await _gateway(server, remote)
    try:
        await a.apply_and_publish(_config(display_name="second"))
        await asyncio.wait_for(remote.event.wait(), 2)
        await asyncio.sleep(0.05)
    finally:
        await a.stop()
        await b.stop()

    assert [c.display_name for c in local.configs] == ["second"]
    assert [c.display_name for c in remote.configs] == ["second"]


async def test_the_receiver_keeps_its_own_worker_url():
    server = fakeredis.FakeServer()
    remote = _Applied()
    a = await _gateway(server, _Applied())
    b = ConfigSync(
        "redis://unused", _CHANNEL, remote, client=fakeredis.FakeAsyncRedis(server=server),
        worker_url_of=lambda model_id: f"http://{model_id}-of-gateway-b:8001",
    )
    b.start()
    await asyncio.wait_for(b.subscribed.wait(), 2)
    try:
        await a.apply_and_publish(_config(worker_url="http://worker-of-gateway-a:8001"))
        await asyncio.wait_for(remote.event.wait(), 2)
    finally:
        await a.stop()
        await b.stop()

    assert remote.configs[0].worker_url == "http://sync-model-of-gateway-b:8001"


async def test_a_worker_url_in_a_message_is_not_applied():
    server = fakeredis.FakeServer()
    applied = _Applied()
    sync = await _gateway(server, applied)
    config = _config(worker_url="http://worker-of-gateway-a:8001").model_dump(mode="json")
    try:
        await fakeredis.FakeAsyncRedis(server=server).publish(
            _CHANNEL, json.dumps({"origin": "a", "config": config})
        )
        await asyncio.wait_for(applied.event.wait(), 2)
    finally:
        await sync.stop()

    assert applied.configs[0].worker_url is None


async def test_a_quiet_subscription_stays_up(monkeypatch):
    monkeypatch.setattr(config_sync, "_HEALTH_CHECK_S", 0.02)
    server = fakeredis.FakeServer()
    applied = _Applied()
    sync = await _gateway(server, applied)
    try:
        await asyncio.sleep(0.15)
        assert sync.subscribed.is_set()
        await fakeredis.FakeAsyncRedis(server=server).publish(
            _CHANNEL, json.dumps({"origin": "a", "config": _config().model_dump(mode="json")})
        )
        await asyncio.wait_for(applied.event.wait(), 2)
    finally:
        await sync.stop()


def test_the_subscription_checks_its_connection(monkeypatch):
    import redis.asyncio as redis_async

    settings: dict = {}
    monkeypatch.setattr(redis_async, "from_url", lambda url, **kwargs: settings.update(kwargs))
    ConfigSync("redis://redis:6379/0", _CHANNEL, _Applied())

    assert settings["health_check_interval"] == config_sync._HEALTH_CHECK_S
    assert settings["socket_connect_timeout"] > 0
    assert settings["socket_timeout"] > 0


async def test_malformed_messages_are_skipped(caplog):
    server = fakeredis.FakeServer()
    applied = _Applied()
    sync = await _gateway(server, applied)
    publisher = fakeredis.FakeAsyncRedis(server=server)
    with caplog.at_level(logging.WARNING, logger="app.services.config_sync"):
        await publisher.publish(_CHANNEL, "not json")
        await publisher.publish(_CHANNEL, json.dumps({"origin": "x", "config": {"id": "bad id"}}))
        await asyncio.sleep(0.1)
    await sync.stop()

    assert applied.configs == []
    assert sum("malformed" in r.message for r in caplog.records) == 2


async def test_publish_without_redis_only_warns(caplog):
    server = fakeredis.FakeServer()
    server.connected = False
    sync = ConfigSync("redis://unused", _CHANNEL, _Applied(), client=fakeredis.FakeAsyncRedis(server=server))
    with caplog.at_level(logging.WARNING, logger="app.services.config_sync"):
        await sync.publish(_config().model_dump(mode="json"))

    assert any("could not relay sync-model" in r.message for r in caplog.records)


async def test_listener_subscribes_once_redis_is_back(monkeypatch):
    monkeypatch.setattr(config_sync, "_MIN_BACKOFF_S", 0.01)
    server = fakeredis.FakeServer()
    server.connected = False
    sync = ConfigSync("redis://unused", _CHANNEL, _Applied(), client=fakeredis.FakeAsyncRedis(server=server))
    sync.start()
    await asyncio.sleep(0.05)
    assert not sync.subscribed.is_set()

    server.connected = True
    await asyncio.wait_for(sync.subscribed.wait(), 2)
    await sync.stop()


@pytest.mark.parametrize("enabled", [False, True])
def test_server_config_reads_config_sync(enabled, tmp_path):
    from app.config import load_server_config

    path = tmp_path / "server.yaml"
    path.write_text(f"config_sync:\n  enabled: {str(enabled).lower()}\n")
    cfg = load_server_config(path)
    assert cfg.config_sync.enabled is enabled
    assert cfg.config_sync.channel == "infergate:config"
