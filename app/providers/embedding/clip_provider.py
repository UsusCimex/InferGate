from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from app.providers.base import MultimodalEmbeddingProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class CLIPProvider(MultimodalEmbeddingProvider):
    """OpenAI CLIP joint text+image embeddings (e.g. clip-vit-base-patch32, 512-d)."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._device = "cpu"

    async def load(self, model_dir: str) -> None:
        from transformers import CLIPModel, CLIPProcessor

        hub_id = self.config.model["hub_id"]
        device = self.config.model.get("device", "cuda")
        logger.info("Loading %s from %s (device=%s)", self.model_id, hub_id, device)

        loop = asyncio.get_running_loop()

        def _load():
            processor = CLIPProcessor.from_pretrained(hub_id, cache_dir=model_dir)
            model = CLIPModel.from_pretrained(hub_id, cache_dir=model_dir)
            try:
                import torch
                resolved = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
            except ImportError:
                resolved = "cpu"
            model = model.to(resolved)
            model.eval()
            return processor, model, resolved

        self._processor, self._model, self._device = await loop.run_in_executor(None, _load)
        self._loaded = True
        logger.info("Loaded %s on %s", self.model_id, self._device)

    async def unload(self) -> None:
        import gc

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
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

    async def embed(self, inputs: list[str], **params: Any) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed_text_sync, inputs)

    def _embed_text_sync(self, inputs: list[str]) -> list[list[float]]:
        import torch

        tok = self._processor(  # type: ignore[misc]
            text=inputs, padding=True, truncation=True, return_tensors="pt",
        )
        tok = {k: v.to(self._device) for k, v in tok.items()}
        with torch.no_grad():
            features = self._model.get_text_features(**tok)  # type: ignore[union-attr]
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().tolist()

    async def embed_image(self, image: bytes, **params: Any) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed_image_sync, image)

    def _embed_image_sync(self, image: bytes) -> list[float]:
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(image)).convert("RGB")
        proc = self._processor(images=img, return_tensors="pt")  # type: ignore[misc]
        proc = {k: v.to(self._device) for k, v in proc.items()}
        with torch.no_grad():
            features = self._model.get_image_features(**proc)  # type: ignore[union-attr]
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()
