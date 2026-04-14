"""Remote providers that proxy requests to worker containers via HTTP."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import httpx

from app.providers.base import ImageProvider, TextProvider, TtsProvider

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=10.0)
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


class _RemoteMixin:
    """Shared HTTP client logic for remote providers. Used via composition."""

    _client: httpx.AsyncClient | None
    _worker_url: str

    def _init_remote(self) -> None:
        self._client = None
        self._worker_url = self.config.worker_url  # type: ignore[attr-defined]

    async def _remote_load(self) -> None:
        """Connect to worker, wait for readiness, signal load."""
        self._client = httpx.AsyncClient(base_url=self._worker_url, timeout=_TIMEOUT)

        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.get("/health")
                if resp.status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            if attempt < _MAX_RETRIES - 1:
                logger.info(
                    "Worker %s not ready, retrying in %.0fs... (%d/%d)",
                    self._worker_url, _RETRY_DELAY, attempt + 1, _MAX_RETRIES,
                )
                await asyncio.sleep(_RETRY_DELAY)
        else:
            raise RuntimeError(
                f"Worker at {self._worker_url} is not reachable after {_MAX_RETRIES} retries"
            )

        try:
            await self._client.post("/load")
        except httpx.HTTPError:
            pass

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


class RemoteTextProvider(TextProvider):
    """Proxies text generation requests to a remote worker."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        _RemoteMixin._init_remote(self)

    async def load(self, model_dir: str) -> None:
        await _RemoteMixin._remote_load(self)

    async def unload(self) -> None:
        await _RemoteMixin._remote_unload(self)

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

    async def synthesize(self, text: str, **params: Any) -> bytes:
        resp = await self._client.post("/synthesize", json={"text": text, **params})
        resp.raise_for_status()
        return resp.content
