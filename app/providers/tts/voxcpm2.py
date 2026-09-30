from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.providers.base import TtsProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)

VOICES_DIR = Path(__file__).with_name("voxcpm2_voices")
_SF_FORMATS = {"mp3": "MP3", "wav": "WAV", "flac": "FLAC", "opus": "OGG"}


@dataclass(frozen=True)
class PresetVoice:
    """A bundled reference clip and the exact transcript spoken in it."""
    audio_path: Path
    transcript: str


def load_preset_voices(directory: Path = VOICES_DIR) -> dict[str, PresetVoice]:
    """Map every `<id>.wav` in `directory` that has an `<id>.txt` transcript to a preset voice."""
    return {
        wav.stem: PresetVoice(wav, wav.with_suffix(".txt").read_text(encoding="utf-8").strip())
        for wav in sorted(directory.glob("*.wav"))
        if wav.with_suffix(".txt").is_file()
    }


@register_provider
class VoxCpm2TtsProvider(TtsProvider):
    """VoxCPM2 TTS: preset narrators cloned from bundled clips, or a voice from an uploaded clip."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._voices = load_preset_voices()
        # One thread for load and synthesis: CUDA graph trees are thread-local and re-record per thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxcpm2")

    async def load(self, model_dir: str) -> None:
        hub_id = self.config.model["hub_id"]
        device = self.config.model.get("device", "cuda")
        optimize = bool(self.config.model.get("optimize", True))
        logger.info(
            "Loading %s from %s (device=%s, optimize=%s, voices=%s)",
            self.model_id, hub_id, device, optimize, ", ".join(self._voices),
        )

        def _load():
            import voxcpm.model.voxcpm2 as voxcpm2_model
            from huggingface_hub import snapshot_download
            from voxcpm import VoxCPM

            local_path = snapshot_download(repo_id=hub_id, cache_dir=model_dir)
            load_mmapped = voxcpm2_model.load_file

            def load_prefetched(path, *args, **kwargs):
                _prefetch(path)
                return load_mmapped(path, *args, **kwargs)

            # Prefetch inside VoxCPM's load: its fp32 CPU build evicts an earlier prefetch from the page cache.
            voxcpm2_model.load_file = load_prefetched
            try:
                return VoxCPM(
                    local_path,
                    zipenhancer_model_path=None,
                    enable_denoiser=False,
                    optimize=optimize,
                    device=device,
                )
            finally:
                voxcpm2_model.load_file = load_mmapped

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(self._executor, _load)
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def unload(self) -> None:
        import gc

        self._model = None

        def _release() -> None:
            gc.collect()
            try:
                import torch
                import torch._dynamo
            except ImportError:
                return
            torch._dynamo.reset()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _release)
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    def resolve_voice(self, voice: str | None) -> PresetVoice:
        """Return the preset for `voice` ("default" or None picks the configured default)."""
        default = str(self.config.model.get("default_params", {}).get("voice", "vox_daniel"))
        name = default if voice in (None, "", "default") else str(voice)
        preset = self._voices.get(name)
        if preset is None:
            raise ValueError(
                f"unknown voice '{name}' for model '{self.model_id}'; "
                f"use one of: {', '.join(self._voices)}"
            )
        return preset

    async def synthesize(self, text: str, **params: Any) -> bytes:
        if self._model is None:
            raise RuntimeError(f"Model {self.model_id} is not loaded properly")

        opts = dict(self.config.model.get("default_params", {}))
        opts.update(params)
        ref_audio = opts.pop("reference_audio", None)
        ref_filename = str(opts.pop("reference_filename", "ref.wav"))
        ref_text = opts.pop("reference_text", None)
        voice = opts.pop("voice", None)
        output_format = str(opts.pop("output_format", "mp3"))
        seed = int(opts.pop("seed", 42))
        target_lufs = float(opts.pop("loudness_lufs", -25.0))
        peak_dbfs = float(opts.pop("peak_dbfs", -1.0))
        generate_kwargs: dict[str, Any] = {
            "cfg_value": float(opts.pop("cfg_value", 2.0)),
            "inference_timesteps": int(opts.pop("inference_timesteps", 10)),
        }

        tmp_path: str | None = None
        if ref_audio is not None:
            suffix = "." + ref_filename.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(ref_audio)
                tmp_path = tmp.name
            generate_kwargs["reference_wav_path"] = tmp_path
            if ref_text:
                generate_kwargs["prompt_wav_path"] = tmp_path
                generate_kwargs["prompt_text"] = _joinable(str(ref_text))
        else:
            preset = self.resolve_voice(voice)
            generate_kwargs["reference_wav_path"] = str(preset.audio_path)
            generate_kwargs["prompt_wav_path"] = str(preset.audio_path)
            generate_kwargs["prompt_text"] = _joinable(preset.transcript)

        model = self._model

        def _run():
            import torch

            torch.manual_seed(seed)
            wav = model.generate(text=text, **generate_kwargs)
            sample_rate = int(model.tts_model.sample_rate)
            wav = normalize_loudness(wav, sample_rate, target_lufs, peak_dbfs)
            return encode_audio(wav, sample_rate, output_format)

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, _run)
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)


def _prefetch(path: str) -> None:
    # mmap page faults over the Docker Desktop 9p bind mount are ~5x slower than one sequential read.
    with open(path, "rb") as f:
        while f.read(64 << 20):
            pass


def _joinable(transcript: str) -> str:
    # VoxCPM2 prepends the prompt transcript to the target text verbatim.
    return transcript if transcript.endswith((" ", "\n")) else transcript + " "


def normalize_loudness(wav, sample_rate: int, target_lufs: float, peak_dbfs: float):
    """Scale `wav` to `target_lufs` integrated loudness, capped at the `peak_dbfs` sample peak."""
    import numpy as np
    import pyloudnorm

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if not wav.size:
        return wav
    # BS.1770 gating needs one full 400 ms block; the padded silence falls below the absolute gate.
    min_len = int(sample_rate * 0.4) + 1
    padded = wav if wav.size >= min_len else np.pad(wav, (0, min_len - wav.size))
    loudness = pyloudnorm.Meter(sample_rate).integrated_loudness(padded)
    if np.isfinite(loudness):
        wav = wav * np.float32(10 ** ((target_lufs - loudness) / 20))
    ceiling = 10 ** (peak_dbfs / 20)
    peak = float(np.max(np.abs(wav)))
    if peak > ceiling:
        wav = wav * np.float32(ceiling / peak)
    return wav


def encode_audio(wav, sample_rate: int, output_format: str) -> bytes:
    """Encode mono float samples as mp3 / wav / flac / opus (ogg) bytes."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format=_SF_FORMATS.get(output_format, "WAV"))
    return buf.getvalue()
