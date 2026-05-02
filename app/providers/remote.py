from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.monitoring import get_request_id
from app.providers.base import (
    AudioEmbeddingProvider,
    ImageProvider,
    ImageUpscaleProvider,
    MultimodalEmbeddingProvider,
    SttProvider,
    TextEmbeddingProvider,
    TextProvider,
    TtsProvider,
    VideoEmbeddingProvider,
)

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Per-endpoint timeouts: load is slow (model JIT), generate is bounded by SLO,
# health/stats are short-poll. Tunable via env without touching code.
_CONNECT_TIMEOUT = _env_float("GATEWAY_REMOTE_CONNECT_TIMEOUT", 5.0)
_LOAD_TIMEOUT = _env_float("GATEWAY_REMOTE_LOAD_TIMEOUT", 1800.0)
_GENERATE_TIMEOUT = _env_float("GATEWAY_REMOTE_GENERATE_TIMEOUT", 300.0)
_QUICK_TIMEOUT = _env_float("GATEWAY_REMOTE_QUICK_TIMEOUT", 5.0)

_MAX_KEEPALIVE = _env_int("GATEWAY_REMOTE_MAX_KEEPALIVE", 20)
_MAX_CONNECTIONS = _env_int("GATEWAY_REMOTE_MAX_CONNECTIONS", 100)

_RETRY_ATTEMPTS = max(1, _env_int("GATEWAY_REMOTE_RETRY_ATTEMPTS", 3))
_RETRY_BASE_BACKOFF = _env_float("GATEWAY_REMOTE_RETRY_BACKOFF", 0.25)


def _generate_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_GENERATE_TIMEOUT, write=10.0, pool=10.0)


def _load_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_LOAD_TIMEOUT, write=10.0, pool=10.0)


def _quick_timeout() -> httpx.Timeout:
    return httpx.Timeout(_QUICK_TIMEOUT)


def _request_id_headers() -> dict[str, str]:
    rid = get_request_id()
    return {"X-Request-ID": rid} if rid else {}


# Errors safe to retry: nothing was sent on the wire, or the server explicitly
# signalled it didn't accept the request body.
_RETRYABLE_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


async def _retry_send(
    method: str,
    url_or_log: str,
    coro_factory,
) -> httpx.Response:
    """Run `coro_factory()` up to N times, retrying only safe transport failures."""
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return await coro_factory()
        except _RETRYABLE_ERRORS as e:
            last_exc = e
            if attempt + 1 >= _RETRY_ATTEMPTS:
                break
            backoff = _RETRY_BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.1)
            logger.warning(
                "Remote %s %s failed (%s) — retrying in %.2fs (attempt %d/%d)",
                method, url_or_log, type(e).__name__, backoff, attempt + 2, _RETRY_ATTEMPTS,
            )
            await asyncio.sleep(backoff)
    assert last_exc is not None
    raise last_exc


class BaseRemoteMixin:
    """Shared HTTP lifecycle for providers that proxy to a worker container."""

    _client: httpx.AsyncClient | None

    def __init__(self, config: Any) -> None:
        super().__init__(config)  # type: ignore[misc]
        self._client = None
        self._worker_url: str = config.worker_url

    def _build_client(self, timeout: httpx.Timeout) -> httpx.AsyncClient:
        limits = httpx.Limits(
            max_keepalive_connections=_MAX_KEEPALIVE,
            max_connections=_MAX_CONNECTIONS,
        )
        return httpx.AsyncClient(
            base_url=self._worker_url,
            timeout=timeout,
            limits=limits,
        )

    async def load(self, model_dir: str) -> None:
        """Connect to the worker and POST /load; raises on either failure."""
        self._client = self._build_client(_generate_timeout())

        httpx_logger = logging.getLogger("httpx")
        prev_level = httpx_logger.level
        httpx_logger.setLevel(logging.WARNING)
        try:
            try:
                resp = await self._client.get(
                    "/health",
                    timeout=_quick_timeout(),
                    headers=_request_id_headers(),
                )
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
                load_resp = await self._client.post(
                    "/load",
                    timeout=_load_timeout(),
                    headers=_request_id_headers(),
                )
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
                await self._client.post(
                    "/unload",
                    timeout=_quick_timeout(),
                    headers=_request_id_headers(),
                )
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
        resp = await self._client.post(
            "/reload",
            json=new_config.model_dump(mode="json"),
            timeout=_load_timeout(),
            headers=_request_id_headers(),
        )
        resp.raise_for_status()
        return resp.json().get("action", "unknown")

    async def get_stats(self) -> dict:
        """GET /stats from the worker; returns {} when the worker is unreachable."""
        if self._client is None:
            return {}
        try:
            resp = await self._client.get("/stats", timeout=_quick_timeout())
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return {}

    async def _post_json(self, path: str, payload: dict) -> httpx.Response:
        client = self._client
        assert client is not None, "remote provider not loaded"
        headers = _request_id_headers()
        resp = await _retry_send(
            "POST", path,
            lambda: client.post(path, json=payload, headers=headers),
        )
        resp.raise_for_status()
        return resp

    async def _post_files(
        self, path: str, *, files: dict, data: dict | None = None
    ) -> httpx.Response:
        client = self._client
        assert client is not None, "remote provider not loaded"
        headers = _request_id_headers()
        # Don't auto-retry multipart uploads — body is consumed once; build per-attempt.
        resp = await client.post(path, files=files, data=data, headers=headers)
        resp.raise_for_status()
        return resp


class RemoteTextProvider(BaseRemoteMixin, TextProvider):
    """Text-generation provider that proxies to a remote worker."""

    async def generate(self, messages: list[dict], **params: Any) -> dict:
        resp = await self._post_json("/generate", {"messages": messages, **params})
        return resp.json()

    async def generate_stream(self, messages: list[dict], **params: Any) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST", "/generate",
            json={"messages": messages, "stream": True, **params},
            headers=_request_id_headers(),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    yield line + "\n"


class RemoteImageProvider(BaseRemoteMixin, ImageProvider):
    """Image-generation provider that proxies to a remote worker."""

    async def generate(self, prompt: str, **params: Any) -> bytes:
        resp = await self._post_json("/generate", {"prompt": prompt, **params})
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
            resp = await self._post_files("/voice-clone", files=files, data=form)
        else:
            resp = await self._post_json("/synthesize", {"text": text, **params})
        return resp.content


class RemoteSttProvider(BaseRemoteMixin, SttProvider):
    """STT provider that proxies to a remote worker via multipart /transcribe."""

    async def transcribe(self, audio: bytes, **params: Any) -> dict:
        filename = str(params.pop("filename", "audio.wav"))
        form: dict[str, str] = {
            k: str(v) for k, v in params.items() if v is not None
        }
        files = {"file": (filename, audio, "application/octet-stream")}
        resp = await self._post_files("/transcribe", files=files, data=form)
        return resp.json()


class RemoteUpscaleProvider(BaseRemoteMixin, ImageUpscaleProvider):
    """Upscale provider that proxies to a remote worker via multipart /upscale."""

    async def upscale(self, image: bytes, **params: Any) -> bytes:
        files = {"file": ("image.png", image, "image/png")}
        resp = await self._post_files("/upscale", files=files)
        return resp.content


class RemoteTextEmbeddingProvider(BaseRemoteMixin, TextEmbeddingProvider):
    """Text-embedding provider that proxies to a remote worker via JSON /embed."""

    async def embed(self, inputs: list[str], **params: Any) -> list[list[float]]:
        resp = await self._post_json("/embed", {"input": inputs, **params})
        return resp.json()["embeddings"]


class RemoteAudioEmbeddingProvider(BaseRemoteMixin, AudioEmbeddingProvider):
    """Audio-embedding provider that proxies to a remote worker via multipart /embed-audio."""

    async def embed(self, audio: bytes, **params: Any) -> list[float]:
        filename = str(params.pop("filename", "audio.wav"))
        files = {"file": (filename, audio, "application/octet-stream")}
        resp = await self._post_files("/embed-audio", files=files)
        return resp.json()["embedding"]


class RemoteMultimodalEmbeddingProvider(BaseRemoteMixin, MultimodalEmbeddingProvider):
    """Multimodal text+image embedding provider over JSON /embed and multipart /embed-image."""

    async def embed(self, inputs: list[str], **params: Any) -> list[list[float]]:
        resp = await self._post_json("/embed", {"input": inputs, **params})
        return resp.json()["embeddings"]

    async def embed_image(self, image: bytes, **params: Any) -> list[float]:
        filename = str(params.pop("filename", "image.jpg"))
        files = {"file": (filename, image, "application/octet-stream")}
        resp = await self._post_files("/embed-image", files=files)
        return resp.json()["embedding"]


class RemoteVideoEmbeddingProvider(BaseRemoteMixin, VideoEmbeddingProvider):
    """Video+text embedding provider over JSON /embed and multipart /embed-video."""

    async def embed(self, inputs: list[str], **params: Any) -> list[list[float]]:
        resp = await self._post_json("/embed", {"input": inputs, **params})
        return resp.json()["embeddings"]

    async def embed_video(self, video: bytes, **params: Any) -> list[float]:
        filename = str(params.pop("filename", "clip.mp4"))
        files = {"file": (filename, video, "application/octet-stream")}
        resp = await self._post_files("/embed-video", files=files)
        return resp.json()["embedding"]
