from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from app.providers.base import TextProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)

SERVER_PORT = 8080


@register_provider
class LlamaCppTextProvider(TextProvider):
    """GGUF text model served by llama.cpp's llama-server, a child process of the worker on localhost."""

    def __init__(self, config):
        super().__init__(config)
        self._process: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None

    async def load(self, model_dir: str) -> None:
        path = await asyncio.to_thread(self.gguf_path, model_dir)
        args = [
            os.environ.get("LLAMA_SERVER", "llama-server"),
            "-m", path,
            "--alias", self.model_id,
            "--host", "127.0.0.1",
            "--port", str(SERVER_PORT),
            "-c", str(self.config.model.get("context_length", 8192)),
            *(str(arg) for arg in self.config.model.get("server_args", [])),
        ]
        logger.info("Starting llama-server for %s: %s", self.model_id, " ".join(args[1:]))
        self._process = await asyncio.create_subprocess_exec(*args)
        self._client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{SERVER_PORT}", timeout=httpx.Timeout(None, connect=5.0)
        )
        try:
            await self._wait_ready(float(self.config.model.get("load_timeout", 900)))
        except Exception:
            await self.unload()
            raise
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    def gguf_path(self, model_dir: str) -> str:
        """The GGUF file from the Hugging Face cache in [model_dir], downloaded when it is not there."""
        repo = self.config.model["hub_id"]
        filename = self.config.model["gguf_file"]
        revision = self.config.model.get("revision", "main")
        cache = Path(model_dir) / f"models--{repo.replace('/', '--')}"
        ref = cache / "refs" / revision
        snapshot = ref.read_text().strip() if ref.is_file() else revision
        cached = cache / "snapshots" / snapshot / filename
        if cached.is_file():
            return str(cached)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo, filename, revision=revision, cache_dir=model_dir)

    async def _wait_ready(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._process is None or self._process.returncode is not None:
                code = None if self._process is None else self._process.returncode
                raise RuntimeError(f"llama-server exited with code {code} while loading {self.model_id}")
            try:
                if (await self._client.get("/health")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(1.0)
        raise TimeoutError(f"llama-server did not load {self.model_id} in {seconds:.0f} s")

    async def unload(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                process.kill()
                await process.wait()
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    def request_body(self, messages: list[dict], params: dict[str, Any], stream: bool) -> dict[str, Any]:
        """OpenAI chat body for llama-server; thinking is set by the server flags, not per request."""
        body = dict(self.config.model.get("default_params", {}))
        body.update(params)
        body.pop("thinking", None)
        response_format = body.pop("response_format", None)
        if response_format:
            body["response_format"] = {"type": response_format}
        body.update(model=self.model_id, messages=messages, stream=stream)
        return body

    async def generate(self, messages: list[dict], **params: Any) -> dict:
        response = await self._client.post(
            "/v1/chat/completions", json=self.request_body(messages, params, stream=False)
        )
        _raise_for_status(response)
        result = response.json()
        result["model"] = self.model_id
        return result

    async def generate_stream(self, messages: list[dict], **params: Any) -> AsyncIterator[str]:
        body = self.request_body(messages, params, stream=True)
        async with self._client.stream("POST", "/v1/chat/completions", json=body) as response:
            if response.status_code >= 400:
                await response.aread()
                _raise_for_status(response)
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    yield line + "\n\n"


def _raise_for_status(response: httpx.Response) -> None:
    """A request llama-server rejects is the caller's error (HTTP 400 from the worker), the rest a server error."""
    if 400 <= response.status_code < 500:
        raise ValueError(f"llama-server: {response.text[:500]}")
    response.raise_for_status()
