from __future__ import annotations

import asyncio
import contextlib
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

        loop = asyncio.get_running_loop()

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

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, gc.collect)
        try:
            import torch
            if torch.cuda.is_available():
                def _cuda_cleanup() -> None:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                await loop.run_in_executor(None, _cuda_cleanup)
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

        # Voice-cloning payload: write reference to a temp file so we can
        # pass it as a path to TTS.synthesize — most Fish-Speech versions
        # expect a filesystem path rather than raw bytes.
        ref_audio = defaults.pop("reference_audio", None)
        ref_filename = str(defaults.pop("reference_filename", "ref.wav"))
        ref_text = defaults.pop("reference_text", None)

        import os
        import tempfile

        ref_path: str | None = None
        if ref_audio is not None:
            suffix = "." + ref_filename.rsplit(".", 1)[-1]
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(ref_audio)
            tmp.close()
            ref_path = tmp.name

        def _run():
            kwargs: dict[str, Any] = {}
            if ref_path is not None:
                kwargs["reference_audio"] = ref_path
            if ref_text:
                kwargs["reference_text"] = ref_text

            if hasattr(self._model, "synthesize"):
                try:
                    return self._model.synthesize(text, **kwargs)
                except TypeError as e:
                    if kwargs:
                        raise ValueError(
                            f"this fish-speech build does not accept voice-cloning "
                            f"kwargs (reference_audio/reference_text): {e}"
                        ) from e
                    return self._model.synthesize(text)
            # Callable-style fallback; can't carry cloning kwargs.
            if kwargs:
                raise ValueError(
                    "fish-speech model is call-style and cannot receive "
                    "reference_audio/reference_text — upgrade fish_speech to "
                    "a release whose TTS exposes a synthesize() method"
                )
            return self._model(text)

        loop = asyncio.get_running_loop()
        try:
            audio_data = await loop.run_in_executor(None, _run)
        finally:
            if ref_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(ref_path)

        import soundfile as sf

        buf = io.BytesIO()
        fmt = {"mp3": "mp3", "wav": "wav", "flac": "flac"}.get(output_format, "wav")
        sample_rate = getattr(self._model, "sample_rate", 44100)
        sf.write(buf, audio_data, sample_rate, format=fmt)
        buf.seek(0)
        return buf.getvalue()
