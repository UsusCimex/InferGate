from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

from app.config import ModelConfig
from app.services.config_watcher import ReloadCallback

logger = logging.getLogger(__name__)

_MIN_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 60.0
# A quiet subscription PINGs Redis this often, so a connection lost without a word is noticed.
_HEALTH_CHECK_S = 30.0


class ConfigSync:
    """Relays model config changes between gateway instances over a Redis pub/sub channel.

    Messages carry the config as this instance parsed it from YAML, without worker_url: a receiver
    keeps its own (`worker_url_of`). An instance never applies its own messages. Pub/sub keeps no
    history: a gateway that was down reads the YAML files when it starts.
    """

    def __init__(
        self,
        url: str,
        channel: str,
        apply: ReloadCallback,
        *,
        client: Any | None = None,
        worker_url_of: Callable[[str], str | None] | None = None,
    ) -> None:
        self._channel = channel
        self._apply = apply
        self._worker_url_of = worker_url_of
        self._instance = uuid.uuid4().hex
        self._owns_client = client is None
        if client is None:
            import redis.asyncio as redis_async

            client = redis_async.from_url(
                url,
                health_check_interval=_HEALTH_CHECK_S,
                socket_connect_timeout=5,
                socket_timeout=10,
                socket_keepalive=True,
            )
        self._client = client
        self._task: asyncio.Task | None = None
        self.subscribed = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._owns_client:
            await self._client.aclose()

    async def apply_and_publish(self, config: ModelConfig) -> None:
        """Apply a config changed on this instance, then relay it to the other gateways."""
        snapshot = config.model_dump(mode="json", exclude={"worker_url"})
        await self._apply(config)
        await self.publish(snapshot)

    async def publish(self, config: dict[str, Any]) -> None:
        from redis.exceptions import RedisError

        message = json.dumps({"origin": self._instance, "config": config})
        try:
            await self._client.publish(self._channel, message)
        except RedisError as e:
            logger.warning("Config sync: could not relay %s: %s", config.get("id"), e)

    async def _listen(self) -> None:
        backoff = _MIN_BACKOFF_S
        while True:
            try:
                async with self._client.pubsub() as pubsub:
                    await pubsub.subscribe(self._channel)
                    self.subscribed.set()
                    backoff = _MIN_BACKOFF_S
                    logger.info("Config sync subscribed to %s", self._channel)
                    while True:
                        # Waiting with a timeout, not listen(): each call may send the health PING.
                        message = await pubsub.get_message(timeout=_HEALTH_CHECK_S)
                        if message is not None and message["type"] == "message":
                            await self._handle(message["data"])
            except Exception as e:
                self.subscribed.clear()
                logger.warning("Config sync: channel lost (%s), retrying in %.0fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_S)

    async def _handle(self, data: bytes | str) -> None:
        try:
            payload = json.loads(data)
            if payload.get("origin") == self._instance:
                return
            fields = {k: v for k, v in payload["config"].items() if k != "worker_url"}
            config = ModelConfig(**fields)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning("Config sync: ignoring a malformed message: %s", e)
            return
        if self._worker_url_of is not None:
            config.worker_url = self._worker_url_of(config.id)
        logger.info("Config sync: applying %s from another gateway", config.id)
        try:
            await self._apply(config)
        except Exception as e:
            logger.error("Config sync: applying %s failed: %s", config.id, e)
