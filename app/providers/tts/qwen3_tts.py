"""Qwen3-TTS multilingual voice-clone provider.

Qwen3-TTS (Alibaba, Apache-2.0, released Jan 2026) speaks 10 languages
including Russian with zero-shot voice cloning from a short reference
clip. The 0.6B-Base variant fits comfortably alongside an 8B LLM and an
image model on a 12GB card (~1.8 GB weights in bfloat16 + activations).

Unlike XTTS-v2, Qwen3-TTS requires BOTH the reference audio *and* its
transcription (``reference_text``) to condition the speaker — dropping
either degrades cloning quality drastically, so the provider raises
instead of silently proceeding.

Uses the upstream ``qwen-tts`` package (the model weights need custom
code and aren't loadable via plain transformers). Flash-attention 2
lowers VRAM but is optional — set ``QWEN3_TTS_06B_ATTN=eager`` in
deploy/.env when flash-attn isn't available (WSL2 without full CUDA
toolchain, Blackwell, etc.).
"""
from __future__ import annotations

import asyncio
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
    """Qwen3-TTS voice-clone TTS (0.6B / 1.7B variants, 10 languages)."""

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

        # Route HF downloads into the shared models/ volume so the
        # ~1.8GB checkpoint survives container restarts.
        os.environ.setdefault("HF_HOME", model_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", model_dir)

        logger.info(
            "Loading %s from %s (device=%s, dtype=%s, attn=%s)",
            self.model_id, hub_id, device, dtype_name, attn_impl,
        )
        loop = asyncio.get_running_loop()

        def _load():
            # Known upstream bug (HF discussion #1 on the model card):
            # Qwen3TTSTokenizer.from_pretrained expects every file in the
            # `speech_tokenizer/` subfolder to be on disk, but AutoModel's
            # default file filter doesn't fetch the subfolder's
            # preprocessor_config.json. Snapshot the full repo first,
            # then hand `Qwen3TTSModel.from_pretrained` a local path.
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
        # speed / voice are accepted by the OpenAI-compatible schema but
        # Qwen3-TTS takes neither — drop silently rather than error.
        defaults.pop("speed", None)
        defaults.pop("voice", None)

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
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        import soundfile as sf

        fmt = {"mp3": "mp3", "wav": "wav", "flac": "flac", "opus": "ogg"}.get(
            output_format, "wav"
        )
        buf = io.BytesIO()
        sf.write(buf, wav_array, int(sample_rate), format=fmt)
        buf.seek(0)
        return buf.getvalue()
