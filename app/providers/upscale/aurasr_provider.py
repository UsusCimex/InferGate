from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import Any

from app.providers.base import ImageUpscaleProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class AuraSrUpscaleProvider(ImageUpscaleProvider):
    """AuraSR (GigaGAN) 4x upscaler: 64 px tiles, a second pass offset by half a tile hides their seams."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None

    async def load(self, model_dir: str) -> None:
        from huggingface_hub import snapshot_download

        hub_id = self.config.model["hub_id"]
        logger.info("Loading %s from %s", self.model_id, hub_id)

        def _load():
            from aura_sr import AuraSR

            local = Path(snapshot_download(
                repo_id=hub_id, cache_dir=model_dir, allow_patterns=["config.json", "model.safetensors"],
            ))
            # A local file keeps AuraSR off the default HF cache; it reads config.json next to it.
            return AuraSR.from_pretrained(str(local / "model.safetensors"))

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, _load)
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def unload(self) -> None:
        import gc

        self._model = None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, gc.collect)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def upscale(self, image: bytes, **_params: Any) -> bytes:
        from PIL import Image, UnidentifiedImageError

        max_side = int(self.config.model.get("max_input_side", 2048))
        seed = int(self.config.model.get("seed", 0))
        model = self._model

        def _run() -> bytes:
            import torch

            try:
                img = Image.open(io.BytesIO(image)).convert("RGB")
            except UnidentifiedImageError as e:
                raise ValueError("image is not a recognised PNG/JPEG") from e
            if max(img.size) > max_side:
                raise ValueError(f"image side {max(img.size)}px exceeds max_input_side={max_side}px")
            # The generator takes a noise vector per tile: a fixed seed gives the same picture each time.
            torch.manual_seed(seed)
            out = model.upscale_4x_overlapped(img)
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return buf.getvalue()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run)
