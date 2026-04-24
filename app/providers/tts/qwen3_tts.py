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
class Qwen3TtsProvider(TtsProvider):
    """Qwen3-TTS voice-clone provider (requires both reference_audio and reference_text)."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._device = "cuda:0"

    async def load(self, model_dir: str) -> None:
        import torch

        hub_id = self.config.model["hub_id"]
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = self.config.model.get("device", default_device)
        dtype_name = self.config.model.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_name)
        attn_impl = self.config.model.get("attn_implementation", "eager")

        os.environ.setdefault("HF_HOME", model_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", model_dir)

        logger.info(
            "Loading %s from %s (device=%s, dtype=%s, attn=%s)",
            self.model_id, hub_id, device, dtype_name, attn_impl,
        )
        loop = asyncio.get_running_loop()

        def _load():
            # snapshot_download is required: from_pretrained's file filter misses
            # speech_tokenizer/preprocessor_config.json that the tokenizer needs.
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(repo_id=hub_id, cache_dir=model_dir)
            from qwen_tts import Qwen3TTSModel
            return Qwen3TTSModel.from_pretrained(
                local_path,
                device_map=device,
                dtype=dtype,
                attn_implementation=attn_impl,
            )

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
        ref_text = defaults.pop("reference_text", None)
        language = str(defaults.pop("language", "English"))
        output_format = str(defaults.pop("output_format", "wav"))
        # Qwen3-TTS takes neither speed nor voice — drop silently.
        for k in ("speed", "voice"):
            defaults.pop(k, None)

        if ref_audio is None or not ref_text:
            raise ValueError(
                "qwen3-tts requires BOTH reference_audio and reference_text — "
                "upload the WAV clip together with the text spoken in it."
            )

        suffix = "." + ref_filename.rsplit(".", 1)[-1]
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(ref_audio)
        tmp.close()

        def _run():
            wavs, sr = self._model.generate_voice_clone(  # type: ignore[union-attr]
                text=text,
                language=language,
                ref_audio=tmp.name,
                ref_text=ref_text,
            )
            return wavs[0], sr

        try:
            loop = asyncio.get_running_loop()
            wav_array, sample_rate = await loop.run_in_executor(None, _run)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp.name)

        import soundfile as sf

        fmt = {"mp3": "mp3", "wav": "wav", "flac": "flac", "opus": "ogg"}.get(
            output_format, "wav"
        )
        buf = io.BytesIO()
        sf.write(buf, wav_array, int(sample_rate), format=fmt)
        buf.seek(0)
        return buf.getvalue()
