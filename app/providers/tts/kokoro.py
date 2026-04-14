from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from app.providers.base import TtsProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class KokoroTtsProvider(TtsProvider):
    """Provider for Kokoro TTS models."""

    def __init__(self, config):
        super().__init__(config)
        self._pipeline = None

    async def load(self, model_dir: str) -> None:
        import kokoro

        hub_id = self.config.model["hub_id"]
        logger.info("Loading %s from %s", self.model_id, hub_id)

        loop = asyncio.get_running_loop()
        self._pipeline = await loop.run_in_executor(
            None, lambda: kokoro.KPipeline(lang_code="a")
        )
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def unload(self) -> None:
        del self._pipeline
        self._pipeline = None
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def synthesize(self, text: str, **params: Any) -> bytes:
        import soundfile as sf

        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        voice = _resolve_voice(defaults.pop("voice", "af_heart"))
        speed = defaults.pop("speed", 1.0)
        output_format = defaults.pop("output_format", "mp3")

        loop = asyncio.get_running_loop()

        # Kokoro generates audio samples
        samples_list = await loop.run_in_executor(
            None,
            lambda: list(self._pipeline(text, voice=voice, speed=speed)),
        )

        # Concatenate all audio chunks
        import numpy as np

        # Pipeline yields (graphemes, phonemes, audio) tuples
        all_audio = np.concatenate([gs[2] for gs in samples_list])

        buf = io.BytesIO()
        sf.write(buf, all_audio, 24000, format=_sf_format(output_format))
        buf.seek(0)
        return buf.getvalue()


# Map OpenAI voice names and common aliases to Kokoro voices
_VOICE_MAP = {
    "default": "af_heart",
    "alloy": "af_alloy",
    "nova": "af_nova",
    "shimmer": "af_bella",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
}


def _resolve_voice(voice: str) -> str:
    """Resolve OpenAI/alias voice names to Kokoro voice IDs."""
    return _VOICE_MAP.get(voice, voice)


def _sf_format(fmt: str) -> str:
    return {"mp3": "mp3", "wav": "wav", "flac": "flac", "opus": "ogg"}.get(fmt, "wav")
