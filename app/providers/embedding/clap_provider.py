from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from app.providers.base import AudioEmbeddingProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)

_TARGET_SR = 48000


@register_provider
class ClapEmbeddingProvider(AudioEmbeddingProvider):
    """LAION CLAP audio-embedding provider."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._device = "cpu"

    async def load(self, model_dir: str) -> None:
        from transformers import ClapModel, ClapProcessor

        hub_id = self.config.model["hub_id"]
        device = self.config.model.get("device", "cuda")
        logger.info("Loading %s from %s (device=%s)", self.model_id, hub_id, device)

        loop = asyncio.get_running_loop()

        def _load():
            processor = ClapProcessor.from_pretrained(hub_id, cache_dir=model_dir)
            model = ClapModel.from_pretrained(hub_id, cache_dir=model_dir)
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

    async def embed(self, audio: bytes, **params: Any) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed_sync, audio)

    def _embed_sync(self, audio: bytes) -> list[float]:
        import librosa
        import torch

        samples, _ = librosa.load(io.BytesIO(audio), sr=_TARGET_SR, mono=True)
        inputs = self._processor(  # type: ignore[misc]
            audios=samples, sampling_rate=_TARGET_SR, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self._model.get_audio_features(**inputs)  # type: ignore[union-attr]
        # L2-normalise so cosine similarity == dot product downstream.
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()
