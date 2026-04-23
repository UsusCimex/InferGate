"""Remote providers that proxy requests to worker containers via HTTP."""
from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.providers.base import (
    ImageProvider,
    ImageUpscaleProvider,
    SttProvider,
    TextProvider,
    TtsProvider,
)

logger = logging.getLogger(__name__)

# Read timeout — long by default because image generation with
# sequential_cpu_offload on consumer GPUs can legitimately take 5-15 minutes
# for the first request (kernel JIT) and 1-3 min thereafter. Tune via
# GATEWAY_REMOTE_READ_TIMEOUT env var if needed.
_READ_TIMEOUT = float(os.environ.get("GATEWAY_REMOTE_READ_TIMEOUT", "1800"))
_TIMEOUT = httpx.Timeout(connect=5.0, read=_READ_TIMEOUT, write=10.0, pool=10.0)


class BaseRemoteMixin:
    """Shared HTTP transport for remote providers.

    Subclasses inherit the full lifecycle (`load`, `unload`, `check_health`,
    `reload`, `get_stats`) and only implement the category-specific
    request/response serialisation (e.g. `generate`, `synthesize`).

    Mixin contract: subclasses must also inherit from a provider base
    (e.g. `TextProvider`) that provides `self.config.worker_url`,
    `self.model_id`, and the `self._loaded` flag.
    """

    _client: httpx.AsyncClient | None

    def __init__(self, config: Any) -> None:
        super().__init__(config)  # type: ignore[misc]
        self._client = None
        self._worker_url: str = config.worker_url

    async def load(self, model_dir: str) -> None:
        """Connect to worker, then POST /load to make it load the model.

        Fails if either step errors: connection refused → RuntimeError
        "not reachable"; /load returning non-2xx (e.g. vLLM OOM on KV
        cache) → RuntimeError with the worker's error body surfaced up
        so ensure_loaded doesn't flip `_loaded` to True on a broken
        worker.
        """
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
        """Disconnect from worker."""
        if self._client:
            with contextlib.suppress(httpx.HTTPError):
                await self._client.post("/unload")
            await self._client.aclose()
            self._client = None
        self._loaded = False  # type: ignore[attr-defined]
        logger.info("Disconnected from worker %s", self._worker_url)

    async def check_health(self) -> bool:
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

    async def reload(self, new_config: Any) -> str:
        """POST new config to worker /reload; returns action ("noop"|"metadata"|"full_reload")."""
        if self._client is None:
            raise RuntimeError(
                f"Worker {self._worker_url} not connected — /reload cannot be delivered"
            )
        resp = await self._client.post("/reload", json=new_config.model_dump(mode="json"))
        resp.raise_for_status()
        return resp.json().get("action", "unknown")

    async def get_stats(self) -> dict:
        """GET /stats from worker — cheap poll for watchdog.

        Returns empty dict if worker unreachable so caller can reason about
        "no data" distinctly from "0 used".
        """
        if self._client is None:
            return {}
        try:
            resp = await self._client.get("/stats", timeout=httpx.Timeout(5.0))
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return {}


class RemoteTextProvider(BaseRemoteMixin, TextProvider):
    """Proxies text generation requests to a remote worker."""

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
    """Proxies image generation requests to a remote worker."""

    async def generate(self, prompt: str, **params: Any) -> bytes:
        resp = await self._client.post("/generate", json={"prompt": prompt, **params})
        resp.raise_for_status()
        return resp.content


class RemoteTtsProvider(BaseRemoteMixin, TtsProvider):
    """Proxies TTS requests to a remote worker."""

    async def synthesize(self, text: str, **params: Any) -> bytes:
        # Voice-cloning branch: reference_audio bytes → multipart to
        # /voice-clone; everything else uses the JSON /synthesize path.
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
    """Proxies speech-to-text requests to a remote worker.

    Unlike the other remote providers that ship JSON params, STT also
    streams audio bytes — we send them as a multipart/form-data so the
    worker can use the same decoding path as a direct client upload.
    """

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
    """Proxies super-resolution requests to a remote worker via multipart."""

    async def upscale(self, image: bytes, **params: Any) -> bytes:
        files = {"file": ("image.png", image, "image/png")}
        resp = await self._client.post("/upscale", files=files)
        resp.raise_for_status()
        return resp.content
