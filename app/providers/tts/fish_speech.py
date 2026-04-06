from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from app.providers.base import TtsProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class FishSpeechTtsProvider(TtsProvider):
    """Provider for Fish Speech / OpenAudio TTS models."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None

    async def load(self, model_dir: str) -> None:
        hub_id = self.config.model["hub_id"]
        logger.info("Loading %s from %s", self.model_id, hub_id)

        loop = asyncio.get_event_loop()

        def _load():
            try:
                from fish_speech.tts.api import TTS

                return TTS(llama_path=hub_id)
            except ImportError:
                try:
                    # Fallback for older fish-speech versions
                    from fish_speech.inference import TTSInference

                    return TTSInference(model_path=hub_id, device="cuda")
                except ImportError:
                    logger.warning(
                        "fish_speech not installed, %s will not be functional",
                        self.model_id,
                    )
                    return None

        self._model = await loop.run_in_executor(None, _load)
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def unload(self) -> None:
        import gc

        del self._model
        self._model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def synthesize(self, text: str, **params: Any) -> bytes:
        if self._model is None:
            raise RuntimeError(f"Model {self.model_id} is not loaded properly")

        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        output_format = defaults.pop("output_format", "wav")
        defaults.pop("voice", None)
        defaults.pop("speed", None)

        loop = asyncio.get_event_loop()

        # Try new API first (OpenAudio S1-mini), fall back to old
        if hasattr(self._model, "synthesize"):
            audio_data = await loop.run_in_executor(
                None, lambda: self._model.synthesize(text)
            )
        else:
            audio_data = await loop.run_in_executor(
                None, lambda: self._model(text)
            )

        import soundfile as sf

        buf = io.BytesIO()
        fmt = {"mp3": "mp3", "wav": "wav", "flac": "flac"}.get(output_format, "wav")
        sample_rate = getattr(self._model, "sample_rate", 44100)
        sf.write(buf, audio_data, sample_rate, format=fmt)
        buf.seek(0)
        return buf.getvalue()
