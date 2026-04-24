from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import tempfile
from typing import Any

from app.providers.base import TtsProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class XttsTtsProvider(TtsProvider):
    """Coqui XTTS-v2 voice-cloning TTS provider (requires reference_audio)."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._device = "cpu"

    async def load(self, model_dir: str) -> None:
        import torch

        hub_id = self.config.model.get("hub_id", "tts_models/multilingual/multi-dataset/xtts_v2")
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = self.config.model.get("device", default_device)

        os.environ["TTS_HOME"] = model_dir
        # Required to skip the interactive Coqui Public Model License prompt at load.
        os.environ["COQUI_TOS_AGREED"] = "1"

        logger.info("Loading %s from %s (device=%s)", self.model_id, hub_id, device)
        loop = asyncio.get_running_loop()

        def _load():
            from TTS.api import TTS

            tts = TTS(hub_id, progress_bar=False)
            return tts.to(device)

        self._model = await loop.run_in_executor(None, _load)
        self._device = device
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

    async def synthesize(self, text: str, **params: Any) -> bytes:
        if self._model is None:
            raise RuntimeError(f"Model {self.model_id} is not loaded properly")

        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        ref_audio = defaults.pop("reference_audio", None)
        ref_filename = str(defaults.pop("reference_filename", "ref.wav"))
        language = str(defaults.pop("language", "en"))
        output_format = str(defaults.pop("output_format", "wav"))
        # XTTS-v2 doesn't use reference_text / speed / voice — drop silently.
        for k in ("reference_text", "speed", "voice"):
            defaults.pop(k, None)

        if ref_audio is None:
            raise ValueError(
                "xtts-v2 requires reference_audio for voice cloning — "
                "use POST /v1/audio/speech/voice-clone"
            )

        suffix = "." + ref_filename.rsplit(".", 1)[-1]
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(ref_audio)
        tmp.close()

        def _run():
            return self._model.tts(  # type: ignore[union-attr]
                text=text,
                speaker_wav=tmp.name,
                language=language,
            )

        try:
            loop = asyncio.get_running_loop()
            wav_array = await loop.run_in_executor(None, _run)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp.name)

        import soundfile as sf

        sample_rate = int(self.config.model.get("sample_rate", 24000))
        fmt = {"mp3": "mp3", "wav": "wav", "flac": "flac"}.get(output_format, "wav")
        buf = io.BytesIO()
        sf.write(buf, wav_array, sample_rate, format=fmt)
        buf.seek(0)
        return buf.getvalue()
