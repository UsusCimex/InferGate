"""Masked non-autoregressive text-to-image provider for MeissonFlow/Meissonic.

Unlike diffusion (iterative noise→image) and autoregressive (token-by-token),
Meissonic generates all image tokens in parallel at each step and unmasks the
most confident ones — a MaskGiT-style iterative decode. 64 steps, 1024×1024.

The Meissonic HF repo ships custom pipeline code that is not registered in
diffusers. We vendor `src/pipeline.py`, `src/transformer.py`, `src/scheduler.py`
from github.com/viiika/Meissonic into the worker at /app/_meissonic via the
Dockerfile POST_INSTALL step, then prepend that path to sys.path at load time.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import io
import logging
import sys
from typing import Any

from app.providers.base import ImageProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)

_GPU_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="meissonic-gpu"
)

# Path inside the worker container where POST_INSTALL clones the Meissonic
# source tree. Matches the POST_INSTALL command in docker-bake.hcl /
# docker-compose.yml.
_VENDOR_PATH = "/app/_meissonic"


@register_provider
class MeissonicImageProvider(ImageProvider):
    """Masked-token T2I via MeissonFlow/Meissonic.

    Accepts standard kwargs: `negative_prompt`, `num_inference_steps`,
    `guidance_scale`, `width`, `height`, `seed`.
    """

    def __init__(self, config):
        super().__init__(config)
        self._pipeline = None

    async def load(self, model_dir: str) -> None:
        import torch

        if _VENDOR_PATH not in sys.path:
            sys.path.insert(0, _VENDOR_PATH)

        try:
            from src.pipeline import Pipeline  # type: ignore[import-not-found]
            from src.scheduler import Scheduler  # type: ignore[import-not-found]
            from src.transformer import Transformer2DModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                f"Meissonic source not found at {_VENDOR_PATH}. "
                "Check POST_INSTALL in docker-bake.hcl / docker-compose.yml "
                "clones github.com/viiika/Meissonic there."
            ) from e

        from diffusers import VQModel
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        hub_id = self.config.model["hub_id"]
        dtype_name = self.config.model.get("torch_dtype", "float16")
        dtype = getattr(torch, dtype_name)

        # Blackwell race guard
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()

        logger.info("Loading %s from %s", self.model_id, hub_id)
        loop = asyncio.get_running_loop()

        def _load():
            common = {"cache_dir": model_dir, "torch_dtype": dtype}
            transformer = Transformer2DModel.from_pretrained(
                hub_id, subfolder="transformer", **common
            )
            vqvae = VQModel.from_pretrained(hub_id, subfolder="vqvae", **common)
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                hub_id, subfolder="text_encoder", **common
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                hub_id, subfolder="tokenizer", cache_dir=model_dir
            )
            scheduler = Scheduler.from_pretrained(
                hub_id, subfolder="scheduler", cache_dir=model_dir
            )
            pipe = Pipeline(
                vqvae=vqvae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
            )
            pipe.to("cuda")
            return pipe

        self._pipeline = await loop.run_in_executor(_GPU_EXECUTOR, _load)
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def unload(self) -> None:
        import gc

        import torch

        self._pipeline = None

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, gc.collect)
        if torch.cuda.is_available():
            def _cleanup() -> None:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            await loop.run_in_executor(None, _cleanup)
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def generate(self, prompt: str, **params: Any) -> bytes:
        import torch

        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        if "size" in defaults:
            size = defaults.pop("size")
            if isinstance(size, str) and "x" in size:
                w, h = size.split("x")
                defaults.setdefault("width", int(w))
                defaults.setdefault("height", int(h))

        for k in ("response_format", "n"):
            defaults.pop(k, None)

        seed = defaults.pop("seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))

        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(
            _GPU_EXECUTOR,
            lambda: self._pipeline(prompt=prompt, **defaults).images[0],
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
