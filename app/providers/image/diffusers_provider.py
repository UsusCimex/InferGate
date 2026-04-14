from __future__ import annotations

import asyncio
import concurrent.futures
import io
import logging
from typing import Any

from app.providers.base import ImageProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)

_GPU_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="gpu-inference"
)


@register_provider
class DiffusersImageProvider(ImageProvider):
    """Universal provider for any diffusers-compatible model.
    Supports FLUX, Stable Diffusion, PixArt, etc.
    The specific model is determined by the YAML config (hub_id).
    """

    def __init__(self, config):
        super().__init__(config)
        self._pipeline = None

    async def load(self, model_dir: str) -> None:
        import torch
        from diffusers import DiffusionPipeline

        hub_id = self.config.model["hub_id"]
        dtype_name = self.config.model.get("torch_dtype", "float16")
        dtype = getattr(torch, dtype_name)

        kwargs: dict[str, Any] = {
            "cache_dir": model_dir,
            "torch_dtype": dtype,
        }
        if variant := self.config.model.get("variant"):
            kwargs["variant"] = variant
        if revision := self.config.model.get("revision"):
            kwargs["revision"] = revision

        # Drop T5 text encoder for SD 3.5 to save ~9.5 GB VRAM
        if self.config.model.get("drop_t5", False):
            kwargs["text_encoder_3"] = None
            kwargs["tokenizer_3"] = None

        cpu_offload = self.config.model.get("cpu_offload", False)

        logger.info("Loading %s from %s", self.model_id, hub_id)
        loop = asyncio.get_running_loop()

        def _load():
            pipe = DiffusionPipeline.from_pretrained(hub_id, **kwargs)
            if cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            return pipe

        self._pipeline = await loop.run_in_executor(_GPU_EXECUTOR, _load)
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def unload(self) -> None:
        import gc

        import torch

        if self._pipeline is not None:
            if hasattr(self._pipeline, "maybe_free_model_hooks"):
                self._pipeline.maybe_free_model_hooks()
            del self._pipeline
            self._pipeline = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def generate(self, prompt: str, **params: Any) -> bytes:
        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        # Parse size string if present
        if "size" in defaults:
            size = defaults.pop("size")
            if isinstance(size, str) and "x" in size:
                w, h = size.split("x")
                defaults.setdefault("width", int(w))
                defaults.setdefault("height", int(h))

        # Remove params not accepted by pipeline
        defaults.pop("response_format", None)
        defaults.pop("n", None)

        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(
            None,
            lambda: self._pipeline(prompt, **defaults).images[0],
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
