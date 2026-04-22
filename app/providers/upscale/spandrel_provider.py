"""Super-resolution provider via spandrel — a unified loader covering
ESRGAN, Real-ESRGAN, SwinIR, DAT, HAT, CodeFormer, and friends.

Chosen over single-architecture packages (realesrgan, basicsr) because
spandrel parses the checkpoint's architecture metadata automatically —
one provider class serves any SR model whose .pth lands on HuggingFace.
YAML selects the model via `hub_id` + `filename`; scale factor is read
off the loaded model, not hard-coded.
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from app.providers.base import ImageUpscaleProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class SpandrelUpscaleProvider(ImageUpscaleProvider):
    """Generic super-resolution wrapper around spandrel.ModelLoader."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._device = "cpu"
        self._dtype = None
        self._scale = 1

    async def load(self, model_dir: str) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from spandrel import ModelLoader

        hub_id = self.config.model["hub_id"]
        filename = self.config.model["filename"]
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = self.config.model.get("device", default_device)
        dtype_name = self.config.model.get(
            "torch_dtype", "float16" if device == "cuda" else "float32"
        )

        logger.info("Loading %s from %s/%s (device=%s, dtype=%s)",
                    self.model_id, hub_id, filename, device, dtype_name)
        path = hf_hub_download(repo_id=hub_id, filename=filename, cache_dir=model_dir)

        def _load():
            m = ModelLoader().load_from_file(path)
            dtype = getattr(torch, dtype_name)
            m.model.to(device=device, dtype=dtype).eval()
            return m, dtype

        loop = asyncio.get_running_loop()
        self._model, self._dtype = await loop.run_in_executor(None, _load)
        self._device = device
        self._scale = int(getattr(self._model, "scale", 1))
        self._loaded = True
        logger.info("Loaded %s (architecture=%s, scale=%dx)",
                    self.model_id,
                    getattr(self._model, "architecture", "?"),
                    self._scale)

    async def unload(self) -> None:
        import gc

        if self._model is not None:
            del self._model
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
        import numpy as np
        import torch
        from PIL import Image, UnidentifiedImageError

        max_side = int(self.config.model.get("max_input_side", 2048))

        def _run() -> bytes:
            try:
                img = Image.open(io.BytesIO(image)).convert("RGB")
            except UnidentifiedImageError as e:
                raise ValueError("image is not a recognised PNG/JPEG") from e
            if max(img.size) > max_side:
                raise ValueError(
                    f"image side {max(img.size)}px exceeds max_input_side={max_side}px — "
                    f"tiling is a follow-up feature"
                )

            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensor = (
                torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                .to(device=self._device, dtype=self._dtype)
            )

            with torch.inference_mode():
                out = self._model.model(tensor)  # type: ignore[union-attr]

            out_arr = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
            out_img = Image.fromarray((out_arr * 255).astype(np.uint8))

            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            return buf.getvalue()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run)
