"""Remote providers that proxy requests to worker containers via HTTP."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import httpx

from app.providers.base import ImageProvider, TextProvider, TtsProvider

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=10.0)


class _RemoteMixin:
    """Shared HTTP client logic for remote providers. Used via composition."""

    _client: httpx.AsyncClient | None
    _worker_url: str

    def _init_remote(self) -> None:
        self._client = None
        self._worker_url = self.config.worker_url  # type: ignore[attr-defined]

    async def _remote_load(self) -> None:
        """Connect to worker — single attempt, fail fast."""
        self._client = httpx.AsyncClient(base_url=self._worker_url, timeout=_TIMEOUT)

        httpx_logger = logging.getLogger("httpx")
        prev_level = httpx_logger.level
        httpx_logger.setLevel(logging.WARNING)
        try:
            try:
                resp = await self._client.get("/health")
                if resp.status_code != 200:
                    raise RuntimeError(f"Worker {self._worker_url} returned status {resp.status_code}")
            except httpx.HTTPError:
                await self._client.aclose()
                self._client = None
                raise RuntimeError(
                    f"Worker at {self._worker_url} is not reachable"
                )

            try:
                await self._client.post("/load")
            except httpx.HTTPError:
                pass
        finally:
            httpx_logger.setLevel(prev_level)

        self._loaded = True  # type: ignore[attr-defined]
        logger.info("Connected to worker %s for %s", self._worker_url, self.model_id)  # type: ignore[attr-defined]

    async def _remote_unload(self) -> None:
        """Disconnect from worker."""
        if self._client:
            try:
                await self._client.post("/unload")
            except httpx.HTTPError:
                pass
            await self._client.aclose()
            self._client = None
        self._loaded = False  # type: ignore[attr-defined]
        logger.info("Disconnected from worker %s", self._worker_url)

    async def _check_health(self) -> bool:
        """Single health probe — used by background monitor."""
        httpx_logger = logging.getLogger("httpx")
        prev_level = httpx_logger.level
        httpx_logger.setLevel(logging.WARNING)
        try:
            client = httpx.AsyncClient(base_url=self._worker_url, timeout=httpx.Timeout(3.0))
            try:
                resp = await client.get("/health")
                return resp.status_code == 200
            finally:
                await client.aclose()
        except httpx.HTTPError:
            return False
        finally:
            httpx_logger.setLevel(prev_level)


class RemoteTextProvider(TextProvider):
    """Proxies text generation requests to a remote worker."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        _RemoteMixin._init_remote(self)

    async def load(self, model_dir: str) -> None:
        await _RemoteMixin._remote_load(self)

    async def unload(self) -> None:
        await _RemoteMixin._remote_unload(self)

    async def check_health(self) -> bool:
        return await _RemoteMixin._check_health(self)

    async def generate(self, messages: list[dict], **params: Any) -> dict:
        resp = await self._client.post("/generate", json={"messages": messages, **params})
        resp.raise_for_status()
        return resp.json()

    async def generate_stream(self, messages: list[dict], **params: Any) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST", "/generate", json={"messages": messages, "stream": True, **params},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    yield line + "\n"


class RemoteImageProvider(ImageProvider):
    """Proxies image generation requests to a remote worker."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        _RemoteMixin._init_remote(self)

    async def load(self, model_dir: str) -> None:
        await _RemoteMixin._remote_load(self)

    async def unload(self) -> None:
        await _RemoteMixin._remote_unload(self)

    async def check_health(self) -> bool:
        return await _RemoteMixin._check_health(self)

    async def generate(self, prompt: str, **params: Any) -> bytes:
        resp = await self._client.post("/generate", json={"prompt": prompt, **params})
        resp.raise_for_status()
        return resp.content


class RemoteTtsProvider(TtsProvider):
    """Proxies TTS requests to a remote worker."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        _RemoteMixin._init_remote(self)

    async def load(self, model_dir: str) -> None:
        await _RemoteMixin._remote_load(self)

    async def unload(self) -> None:
        await _RemoteMixin._remote_unload(self)

    async def check_health(self) -> bool:
        return await _RemoteMixin._check_health(self)

    async def synthesize(self, text: str, **params: Any) -> bytes:
        resp = await self._client.post("/synthesize", json={"text": text, **params})
        resp.raise_for_status()
        return resp.content
