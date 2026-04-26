from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.providers.base import TextEmbeddingProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class SentenceTransformerEmbeddingProvider(TextEmbeddingProvider):
    """sentence-transformers text-embedding provider (e.g. multilingual-e5)."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None

    async def load(self, model_dir: str) -> None:
        from sentence_transformers import SentenceTransformer

        hub_id = self.config.model["hub_id"]
        device = self.config.model.get("device", "cuda")
        logger.info("Loading %s from %s (device=%s)", self.model_id, hub_id, device)

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(hub_id, device=device, cache_folder=model_dir),
        )
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

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

    async def embed(self, inputs: list[str], **params: Any) -> list[list[float]]:
        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)
        normalize = bool(defaults.get("normalize", True))

        loop = asyncio.get_running_loop()
        vecs = await loop.run_in_executor(
            None,
            lambda: self._model.encode(  # type: ignore[union-attr]
                inputs,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            ),
        )
        return vecs.tolist()
