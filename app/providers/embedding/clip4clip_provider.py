from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from typing import Any

from app.providers.base import VideoEmbeddingProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class CLIP4ClipProvider(VideoEmbeddingProvider):
    """CLIP4Clip joint text+video embeddings — 12 evenly-spaced frames, mean-pooled into 512-d."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._num_frames = int(config.model.get("num_frames", 12))
        self._fallback_processor_id = config.model.get(
            "fallback_processor_hub_id", "openai/clip-vit-base-patch32"
        )

    async def load(self, model_dir: str) -> None:
        from transformers import CLIPModel, CLIPProcessor

        hub_id = self.config.model["hub_id"]
        device = self.config.model.get("device", "cuda")
        logger.info("Loading %s from %s (device=%s)", self.model_id, hub_id, device)

        loop = asyncio.get_running_loop()

        def _load():
            try:
                processor = CLIPProcessor.from_pretrained(hub_id, cache_dir=model_dir)
            except Exception as e:
                # CLIP4Clip checkpoints sometimes ship only model weights without a processor.
                logger.info(
                    "Processor missing on %s (%s) — falling back to %s",
                    hub_id, e, self._fallback_processor_id,
                )
                processor = CLIPProcessor.from_pretrained(
                    self._fallback_processor_id, cache_dir=model_dir,
                )
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
        logger.info("Loaded %s on %s (frames=%d)", self.model_id, self._device, self._num_frames)

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

    async def embed_video(self, video: bytes, **params: Any) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed_video_sync, video)

    def _embed_video_sync(self, video: bytes) -> list[float]:
        import torch
        from PIL import Image

        frames = self._decode_frames(video, self._num_frames)
        if not frames:
            raise ValueError("could not decode any frames from the uploaded video")
        pil_frames = [Image.fromarray(f).convert("RGB") for f in frames]
        proc = self._processor(images=pil_frames, return_tensors="pt")  # type: ignore[misc]
        proc = {k: v.to(self._device) for k, v in proc.items()}
        with torch.no_grad():
            # (N, 512) per-frame features, then mean-pool into a single 512-d vector.
            per_frame = self._model.get_image_features(**proc)  # type: ignore[union-attr]
        per_frame = per_frame / per_frame.norm(p=2, dim=-1, keepdim=True)
        pooled = per_frame.mean(dim=0)
        pooled = pooled / pooled.norm(p=2)
        return pooled.cpu().tolist()

    @staticmethod
    def _decode_frames(video: bytes, num_frames: int):
        """Extract `num_frames` evenly-spaced RGB frames from a raw video clip."""
        import cv2
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video)
            path = f.name
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return []
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            indices = np.linspace(0, max(0, total - 1), num=num_frames, dtype=int)
            out = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            return out
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)
