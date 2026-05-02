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
from app.providers._remote_protocol import (
    JsonEndpoint,
    MultipartEndpoint,
    call_json,
    call_multipart,
)
from app.providers.base import (
    AudioEmbeddingProvider,
    BaseProvider,
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


_CONNECT_TIMEOUT = _env_float("GATEWAY_REMOTE_CONNECT_TIMEOUT", 5.0)
_LOAD_TIMEOUT = _env_float("GATEWAY_REMOTE_LOAD_TIMEOUT", 1800.0)
_GENERATE_TIMEOUT = _env_float("GATEWAY_REMOTE_GENERATE_TIMEOUT", 300.0)
_QUICK_TIMEOUT = _env_float("GATEWAY_REMOTE_QUICK_TIMEOUT", 5.0)

_MAX_KEEPALIVE = _env_int("GATEWAY_REMOTE_MAX_KEEPALIVE", 20)
_MAX_CONNECTIONS = _env_int("GATEWAY_REMOTE_MAX_CONNECTIONS", 100)

_RETRY_ATTEMPTS = max(1, _env_int("GATEWAY_REMOTE_RETRY_ATTEMPTS", 3))
_RETRY_BASE_BACKOFF = _env_float("GATEWAY_REMOTE_RETRY_BACKOFF", 0.25)

# Stepped backoff: small models (<=0.5s) avoid the full 2s overhead, long loads
# settle into 2s polls. Override via GATEWAY_REMOTE_LOAD_POLL_BACKOFF="0.5,1,2,2,2".
_DEFAULT_POLL_BACKOFF = "0.5,1.0,2.0,2.0,2.0"


def _parse_poll_backoff() -> list[float]:
    raw = os.environ.get("GATEWAY_REMOTE_LOAD_POLL_BACKOFF", _DEFAULT_POLL_BACKOFF)
    out: list[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            v = float(piece)
            if v > 0:
                out.append(v)
        except ValueError:
            continue
    return out or [0.5, 1.0, 2.0, 2.0, 2.0]


_LOAD_POLL_BACKOFF = _parse_poll_backoff()


def _generate_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_GENERATE_TIMEOUT, write=10.0, pool=10.0)


def _load_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_LOAD_TIMEOUT, write=10.0, pool=10.0)


def _quick_timeout() -> httpx.Timeout:
    return httpx.Timeout(_QUICK_TIMEOUT)


def _request_id_headers() -> dict[str, str]:
    rid = get_request_id()
    return {"X-Request-ID": rid} if rid else {}


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


# Endpoint registry: declarative spec of every worker endpoint exposed by the gateway.
# Adding a new category-only requires updating this map and adding a thin subclass below.
GENERATE_TEXT = JsonEndpoint(path="/generate", payload_key="messages")
GENERATE_IMAGE = JsonEndpoint(path="/generate", payload_key="prompt", response_kind="bytes")
SYNTHESIZE_TTS = JsonEndpoint(path="/synthesize", payload_key="text", response_kind="bytes")
EMBED_TEXT = JsonEndpoint(
    path="/embed", payload_key="input",
    response_kind="json_field", response_field="embeddings",
)
TRANSCRIBE = MultipartEndpoint(path="/transcribe", default_filename="audio.wav")
UPSCALE = MultipartEndpoint(
    path="/upscale", default_filename="image.png",
    content_type="image/png", response_kind="bytes",
)
EMBED_AUDIO = MultipartEndpoint(
    path="/embed-audio", default_filename="audio.wav",
    response_kind="json_field", response_field="embedding",
)
EMBED_IMAGE = MultipartEndpoint(
    path="/embed-image", default_filename="image.jpg",
    response_kind="json_field", response_field="embedding",
)
EMBED_VIDEO = MultipartEndpoint(
    path="/embed-video", default_filename="clip.mp4",
    response_kind="json_field", response_field="embedding",
)


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
        """Connect to the worker and ensure the model is loaded.

        Worker contract:
        - POST /load returns 200 → ready immediately (legacy worker fast-path).
        - POST /load returns 202 → background load running; we poll GET /load/status
          until "ready"/"failed" or `_LOAD_TIMEOUT` deadline.
        """
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
            except httpx.HTTPError as e:
                await self._client.aclose()
                self._client = None
                raise RuntimeError(
                    f"Worker {self._worker_url} /load call failed: {e}"
                ) from e

            if load_resp.status_code == 202:
                try:
                    await self._poll_load_status(load_resp)
                except BaseException:
                    # Cancel/timeout/failure → release the httpx pool slot.
                    await self._client.aclose()
                    self._client = None
                    raise
            elif 200 <= load_resp.status_code < 300:
                # Legacy worker — synchronous /load returned success.
                pass
            else:
                body = load_resp.text[:500]
                await self._client.aclose()
                self._client = None
                raise RuntimeError(
                    f"Worker {self._worker_url} rejected /load "
                    f"(status {load_resp.status_code}): {body}"
                )
        finally:
            httpx_logger.setLevel(prev_level)

        self._loaded = True  # type: ignore[attr-defined]
        logger.info(
            "Connected to worker %s for %s",
            self._worker_url, self.model_id,  # type: ignore[attr-defined]
        )

    async def _poll_load_status(self, initial_resp: httpx.Response) -> None:
        """Poll GET /load/status until ready/failed or _LOAD_TIMEOUT elapses."""
        client = self._client
        assert client is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _LOAD_TIMEOUT
        backoff = list(_LOAD_POLL_BACKOFF)

        # Initial 202 body may already say "ready" if the worker raced us.
        try:
            body = initial_resp.json()
        except ValueError:
            body = {}
        state = body.get("load_state") or body
        if state.get("status") == "ready":
            return
        if state.get("status") == "failed":
            raise RuntimeError(
                f"Worker {self._worker_url} load failed: {state.get('error', 'unknown')}"
            )

        idx = 0
        while True:
            now = loop.time()
            if now >= deadline:
                raise RuntimeError(
                    f"Worker {self._worker_url} did not become ready within "
                    f"{_LOAD_TIMEOUT}s (last status: {state.get('status')!r})"
                )
            sleep_for = backoff[min(idx, len(backoff) - 1)]
            sleep_for = min(sleep_for, deadline - now)
            await asyncio.sleep(sleep_for)
            idx += 1

            try:
                resp = await client.get(
                    "/load/status",
                    timeout=_quick_timeout(),
                    headers=_request_id_headers(),
                )
            except httpx.HTTPError as e:
                # Transient — keep polling until the deadline.
                logger.debug("Poll /load/status failed (transient): %s", e)
                continue

            if resp.status_code == 404:
                # Legacy worker without /load/status — accept the original 202 as ready.
                logger.info(
                    "Worker %s lacks /load/status — assuming ready (legacy contract)",
                    self._worker_url,
                )
                return

            if resp.status_code != 200:
                logger.debug(
                    "Poll /load/status returned %d: %s",
                    resp.status_code, resp.text[:200],
                )
                continue

            try:
                state = resp.json()
            except ValueError:
                continue

            status = state.get("status")
            if status == "ready":
                return
            if status == "failed":
                raise RuntimeError(
                    f"Worker {self._worker_url} load failed: "
                    f"{state.get('error', 'unknown')}"
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

    def _client_required(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(f"Remote provider {self._worker_url} is not loaded")
        return self._client

    async def _call_json(self, endpoint: JsonEndpoint, primary: Any, **extra: Any) -> Any:
        async def _send(factory):
            return await _retry_send("POST", endpoint.path, factory)

        return await call_json(
            self._client_required(), endpoint, primary,
            extra=extra, headers=_request_id_headers(), send=_send,
        )

    async def _call_multipart(
        self, endpoint: MultipartEndpoint, file_bytes: bytes,
        filename: str | None = None, form: dict[str, str] | None = None,
    ) -> Any:
        return await call_multipart(
            self._client_required(), endpoint, file_bytes,
            filename=filename, form=form, headers=_request_id_headers(),
        )


class RemoteTextProvider(BaseRemoteMixin, TextProvider):
    """Text-generation provider that proxies to a remote worker."""

    async def generate(self, messages: list[dict], **params: Any) -> dict:
        return await self._call_json(GENERATE_TEXT, messages, **params)

    async def generate_stream(self, messages: list[dict], **params: Any) -> AsyncIterator[str]:
        client = self._client_required()
        async with client.stream(
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
        return await self._call_json(GENERATE_IMAGE, prompt, **params)


class RemoteTtsProvider(BaseRemoteMixin, TtsProvider):
    """TTS provider that proxies to a remote worker (JSON or multipart depending on params)."""

    async def synthesize(self, text: str, **params: Any) -> bytes:
        ref = params.pop("reference_audio", None)
        if ref is None:
            return await self._call_json(SYNTHESIZE_TTS, text, **params)

        # Voice-clone path: dedicated /voice-clone endpoint, multipart payload.
        filename = str(params.pop("reference_filename", "ref.wav"))
        form: dict[str, str] = {"input": text}
        for k, v in params.items():
            if v is not None:
                form[k] = str(v)
        clone_endpoint = MultipartEndpoint(
            path="/voice-clone",
            file_field="reference_audio",
            default_filename="ref.wav",
            response_kind="bytes",
        )
        return await self._call_multipart(clone_endpoint, ref, filename=filename, form=form)


class RemoteSttProvider(BaseRemoteMixin, SttProvider):
    """STT provider that proxies to a remote worker via multipart /transcribe."""

    async def transcribe(self, audio: bytes, **params: Any) -> dict:
        filename = str(params.pop("filename", TRANSCRIBE.default_filename))
        form = {k: str(v) for k, v in params.items() if v is not None}
        return await self._call_multipart(TRANSCRIBE, audio, filename=filename, form=form)


class RemoteUpscaleProvider(BaseRemoteMixin, ImageUpscaleProvider):
    """Upscale provider that proxies to a remote worker via multipart /upscale."""

    async def upscale(self, image: bytes, **params: Any) -> bytes:
        return await self._call_multipart(UPSCALE, image)


class RemoteTextEmbeddingProvider(BaseRemoteMixin, TextEmbeddingProvider):
    """Text-embedding provider that proxies to a remote worker via JSON /embed."""

    async def embed(self, inputs: list[str], **params: Any) -> list[list[float]]:
        return await self._call_json(EMBED_TEXT, inputs, **params)


class RemoteAudioEmbeddingProvider(BaseRemoteMixin, AudioEmbeddingProvider):
    """Audio-embedding provider that proxies to a remote worker via multipart /embed-audio."""

    async def embed(self, audio: bytes, **params: Any) -> list[float]:
        filename = params.pop("filename", None)
        return await self._call_multipart(EMBED_AUDIO, audio, filename=filename)


class RemoteMultimodalEmbeddingProvider(BaseRemoteMixin, MultimodalEmbeddingProvider):
    """Multimodal text+image embedding provider over JSON /embed and multipart /embed-image."""

    async def embed(self, inputs: list[str], **params: Any) -> list[list[float]]:
        return await self._call_json(EMBED_TEXT, inputs, **params)

    async def embed_image(self, image: bytes, **params: Any) -> list[float]:
        filename = params.pop("filename", None)
        return await self._call_multipart(EMBED_IMAGE, image, filename=filename)


class RemoteVideoEmbeddingProvider(BaseRemoteMixin, VideoEmbeddingProvider):
    """Video+text embedding provider over JSON /embed and multipart /embed-video."""

    async def embed(self, inputs: list[str], **params: Any) -> list[list[float]]:
        return await self._call_json(EMBED_TEXT, inputs, **params)

    async def embed_video(self, video: bytes, **params: Any) -> list[float]:
        filename = params.pop("filename", None)
        return await self._call_multipart(EMBED_VIDEO, video, filename=filename)


# Single source of truth for category → RemoteProvider class. Used by the
# ProviderManager when resolving worker_url-backed configs.
CATEGORY_REGISTRY: dict[str, type[BaseProvider]] = {
    "text": RemoteTextProvider,
    "image": RemoteImageProvider,
    "tts": RemoteTtsProvider,
    "stt": RemoteSttProvider,
    "upscale": RemoteUpscaleProvider,
    "embedding-text": RemoteTextEmbeddingProvider,
    "embedding-audio": RemoteAudioEmbeddingProvider,
    "embedding-multimodal": RemoteMultimodalEmbeddingProvider,
    "embedding-video": RemoteVideoEmbeddingProvider,
}


def remote_provider_for(category: str) -> type[BaseProvider]:
    """Return the RemoteProvider class registered for `category` or raise ValueError."""
    cls = CATEGORY_REGISTRY.get(category)
    if cls is None:
        raise ValueError(f"No remote provider for category '{category}'")
    return cls
