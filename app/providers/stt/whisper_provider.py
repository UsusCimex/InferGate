from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from typing import Any

from app.providers.base import SttProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class WhisperProvider(SttProvider):
    """faster-whisper ASR provider."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._model_dir: str | None = None

    async def load(self, model_dir: str) -> None:
        from faster_whisper import WhisperModel

        self._model_dir = model_dir
        hub_id = self.config.model["hub_id"]
        device = self.config.model.get("device", "cuda")
        compute_type = self.config.model.get("compute_type", "float16")
        cpu_threads = int(self.config.model.get("cpu_threads", 0))

        logger.info(
            "Loading %s from %s (device=%s, compute_type=%s)",
            self.model_id, hub_id, device, compute_type,
        )
        loop = asyncio.get_running_loop()

        def _load():
            return WhisperModel(
                hub_id,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                download_root=model_dir,
            )

        self._model = await loop.run_in_executor(None, _load)
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

    async def transcribe(self, audio: bytes, **params: Any) -> dict:
        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        language = defaults.get("language") or None
        initial_prompt = defaults.get("prompt") or None
        temperature = float(defaults.get("temperature", 0.0))
        beam_size = int(defaults.get("beam_size", 5))
        vad_filter = bool(defaults.get("vad_filter", False))
        response_format = str(defaults.get("response_format", "json"))

        # faster-whisper reads from disk; the ABC contract is bytes-in.
        tmp = tempfile.NamedTemporaryFile(
            suffix="." + str(defaults.get("filename", "audio.wav")).rsplit(".", 1)[-1],
            delete=False,
        )
        tmp.write(audio)
        tmp.close()

        def _run():
            segments_iter, info = self._model.transcribe(  # type: ignore[union-attr]
                tmp.name,
                language=language,
                initial_prompt=initial_prompt,
                temperature=temperature,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
            # Materialise the lazy generator before we unlink the tmp file.
            return list(segments_iter), info

        try:
            loop = asyncio.get_running_loop()
            segments, info = await loop.run_in_executor(None, _run)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp.name)

        text = "".join(s.text for s in segments).strip()
        result: dict[str, Any] = {"text": text}
        if response_format == "verbose_json":
            result["language"] = info.language
            result["duration"] = info.duration
            result["segments"] = [
                {"id": i, "start": float(s.start), "end": float(s.end), "text": s.text}
                for i, s in enumerate(segments)
            ]
        return result
