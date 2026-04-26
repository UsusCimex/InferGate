from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.providers.base import (
    AudioEmbeddingProvider,
    ImageProvider,
    ImageUpscaleProvider,
    MultimodalEmbeddingProvider,
    SttProvider,
    TextEmbeddingProvider,
    TextProvider,
    TtsProvider,
)

logger = logging.getLogger(__name__)

# First-load JITs on consumer GPUs can run 5–15 min; tune via GATEWAY_REMOTE_READ_TIMEOUT.
_READ_TIMEOUT = float(os.environ.get("GATEWAY_REMOTE_READ_TIMEOUT", "1800"))
_TIMEOUT = httpx.Timeout(connect=5.0, read=_READ_TIMEOUT, write=10.0, pool=10.0)


class BaseRemoteMixin:
    """Shared HTTP lifecycle for providers that proxy to a worker container."""

    _client: httpx.AsyncClient | None

    def __init__(self, config: Any) -> None:
        super().__init__(config)  # type: ignore[misc]
        self._client = None
        self._worker_url: str = config.worker_url

    async def load(self, model_dir: str) -> None:
        """Connect to the worker and POST /load; raises on either failure."""
        self._client = httpx.AsyncClient(base_url=self._worker_url, timeout=_TIMEOUT)

        httpx_logger = logging.getLogger("httpx")
        prev_level = httpx_logger.level
        httpx_logger.setLevel(logging.WARNING)
        try:
            try:
                resp = await self._client.get("/health")
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Worker {self._worker_url} returned status {resp.status_code}"
                    )
            except httpx.HTTPError as e:
                await self._client.aclose()
                self._client = None
                raise RuntimeError(
                    f"Worker at {self._worker_url} is not reachable"
                ) from e

            try:
                load_resp = await self._client.post("/load")
                load_resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = e.response.text[:500] if e.response is not None else ""
                await self._client.aclose()
                self._client = None
                raise RuntimeError(
                    f"Worker {self._worker_url} rejected /load "
                    f"(status {e.response.status_code}): {body}"
                ) from e
            except httpx.HTTPError as e:
                await self._client.aclose()
                self._client = None
                raise RuntimeError(
                    f"Worker {self._worker_url} /load call failed: {e}"
                ) from e
        finally:
            httpx_logger.setLevel(prev_level)

        self._loaded = True  # type: ignore[attr-defined]
        logger.info(
            "Connected to worker %s for %s",
            self._worker_url, self.model_id,  # type: ignore[attr-defined]
        )

    async def unload(self) -> None:
        """Best-effort /unload + close the HTTP client."""
        if self._client:
            with contextlib.suppress(httpx.HTTPError):
                await self._client.post("/unload")
            await self._client.aclose()
            self._client = None
        self._loaded = False  # type: ignore[attr-defined]
        logger.info("Disconnected from worker %s", self._worker_url)

    async def check_health(self) -> bool:
        """One-shot /health probe used by the background monitor."""
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

    async def reload(self, new_config: Any) -> str:
        """POST a new config to the worker's /reload; returns its action string."""
        if self._client is None:
            raise RuntimeError(
                f"Worker {self._worker_url} not connected — /reload cannot be delivered"
            )
        resp = await self._client.post("/reload", json=new_config.model_dump(mode="json"))
        resp.raise_for_status()
        return resp.json().get("action", "unknown")

    async def get_stats(self) -> dict:
        """GET /stats from the worker; returns {} when the worker is unreachable."""
        if self._client is None:
            return {}
        try:
            resp = await self._client.get("/stats", timeout=httpx.Timeout(5.0))
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return {}


class RemoteTextProvider(BaseRemoteMixin, TextProvider):
    """Text-generation provider that proxies to a remote worker."""

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


class RemoteImageProvider(BaseRemoteMixin, ImageProvider):
    """Image-generation provider that proxies to a remote worker."""

    async def generate(self, prompt: str, **params: Any) -> bytes:
        resp = await self._client.post("/generate", json={"prompt": prompt, **params})
        resp.raise_for_status()
        return resp.content


class RemoteTtsProvider(BaseRemoteMixin, TtsProvider):
    """TTS provider that proxies to a remote worker (JSON or multipart depending on params)."""

    async def synthesize(self, text: str, **params: Any) -> bytes:
        # reference_audio bytes → multipart to /voice-clone; otherwise JSON /synthesize.
        ref = params.pop("reference_audio", None)
        if ref is not None:
            filename = str(params.pop("reference_filename", "ref.wav"))
            form: dict[str, str] = {"input": text}
            for k, v in params.items():
                if v is not None:
                    form[k] = str(v)
            files = {"reference_audio": (filename, ref, "application/octet-stream")}
            resp = await self._client.post("/voice-clone", files=files, data=form)
        else:
            resp = await self._client.post("/synthesize", json={"text": text, **params})
        resp.raise_for_status()
        return resp.content


class RemoteSttProvider(BaseRemoteMixin, SttProvider):
    """STT provider that proxies to a remote worker via multipart /transcribe."""

    async def transcribe(self, audio: bytes, **params: Any) -> dict:
        filename = str(params.pop("filename", "audio.wav"))
        form: dict[str, str] = {
            k: str(v) for k, v in params.items() if v is not None
        }
        files = {"file": (filename, audio, "application/octet-stream")}
        resp = await self._client.post("/transcribe", files=files, data=form)
        resp.raise_for_status()
        return resp.json()


class RemoteUpscaleProvider(BaseRemoteMixin, ImageUpscaleProvider):
    """Upscale provider that proxies to a remote worker via multipart /upscale."""

    async def upscale(self, image: bytes, **params: Any) -> bytes:
        files = {"file": ("image.png", image, "image/png")}
        resp = await self._client.post("/upscale", files=files)
        resp.raise_for_status()
        return resp.content


class RemoteTextEmbeddingProvider(BaseRemoteMixin, TextEmbeddingProvider):
    """Text-embedding provider that proxies to a remote worker via JSON /embed."""

    async def embed(self, inputs: list[str], **params: Any) -> list[list[float]]:
        resp = await self._client.post("/embed", json={"input": inputs, **params})
        resp.raise_for_status()
        return resp.json()["embeddings"]


class RemoteAudioEmbeddingProvider(BaseRemoteMixin, AudioEmbeddingProvider):
    """Audio-embedding provider that proxies to a remote worker via multipart /embed-audio."""

    async def embed(self, audio: bytes, **params: Any) -> list[float]:
        filename = str(params.pop("filename", "audio.wav"))
        files = {"file": (filename, audio, "application/octet-stream")}
        resp = await self._client.post("/embed-audio", files=files)
        resp.raise_for_status()
        return resp.json()["embedding"]


class RemoteMultimodalEmbeddingProvider(BaseRemoteMixin, MultimodalEmbeddingProvider):
    """Multimodal text+image embedding provider over JSON /embed and multipart /embed-image."""

    async def embed(self, inputs: list[str], **params: Any) -> list[list[float]]:
        resp = await self._client.post("/embed", json={"input": inputs, **params})
        resp.raise_for_status()
        return resp.json()["embeddings"]

    async def embed_image(self, image: bytes, **params: Any) -> list[float]:
        filename = str(params.pop("filename", "image.jpg"))
        files = {"file": (filename, image, "application/octet-stream")}
        resp = await self._client.post("/embed-image", files=files)
        resp.raise_for_status()
        return resp.json()["embedding"]
