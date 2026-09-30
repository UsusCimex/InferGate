"""VoxCpm2TtsProvider against a fake VoxCPM model: presets, cloning, seeding, loudness, formats."""
from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pyloudnorm = pytest.importorskip("pyloudnorm")

from app.config import load_single_model_config  # noqa: E402
from app.providers.tts.voxcpm2 import (  # noqa: E402
    VOICES_DIR,
    VoxCpm2TtsProvider,
    load_preset_voices,
    normalize_loudness,
)

_SR = 48000
_CONFIG = Path(__file__).resolve().parents[1] / "config" / "models" / "voxcpm2.yaml"


def _tone(seconds: float = 1.0, amplitude: float = 0.02):
    t = np.arange(int(_SR * seconds), dtype=np.float32) / _SR
    return amplitude * np.sin(2 * np.pi * 220 * t)


class _FakeVoxCpm:
    def __init__(self):
        self.calls: list[dict] = []
        self.tts_model = types.SimpleNamespace(sample_rate=_SR)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        path = kwargs.get("reference_wav_path")
        kwargs["_reference_existed"] = path is not None and os.path.exists(path)
        return _tone()


@pytest.fixture
def seeds(monkeypatch):
    recorded: list[int] = []
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = recorded.append
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return recorded


@pytest.fixture
def provider(seeds):
    provider = VoxCpm2TtsProvider(load_single_model_config(str(_CONFIG)))
    provider._model = _FakeVoxCpm()
    provider._loaded = True
    return provider


def _removed(path: str) -> bool:
    return not os.path.exists(path)


def _lufs(audio: bytes) -> tuple[float, float]:
    wav, rate = sf.read(io.BytesIO(audio), dtype="float32")
    return pyloudnorm.Meter(rate).integrated_loudness(wav), 20 * np.log10(np.max(np.abs(wav)))


def test_bundled_presets_match_config():
    config = load_single_model_config(str(_CONFIG))
    presets = load_preset_voices()
    assert list(presets) == sorted(config.capabilities.voices)
    assert config.model["default_params"]["voice"] in presets
    for preset in presets.values():
        info = sf.info(str(preset.audio_path))
        assert (info.samplerate, info.channels) == (16000, 1)
        assert 6.0 <= info.duration <= 12.0
        assert preset.transcript.endswith(".")


@pytest.mark.asyncio
async def test_default_voice_clones_the_default_preset(provider, seeds):
    await provider.synthesize("Hello there.", voice="default")
    call = provider._model.calls[-1]
    clip = str(VOICES_DIR / "vox_daniel.wav")
    assert call["prompt_wav_path"] == clip
    assert call["reference_wav_path"] == clip
    assert call["prompt_text"] == load_preset_voices()["vox_daniel"].transcript + " "
    assert call["text"] == "Hello there."
    assert seeds == [42]


@pytest.mark.asyncio
async def test_named_preset_with_request_seed(provider, seeds):
    await provider.synthesize("Hi.", voice="vox_lily", seed=7)
    assert provider._model.calls[-1]["prompt_wav_path"] == str(VOICES_DIR / "vox_lily.wav")
    assert seeds == [7]


@pytest.mark.asyncio
async def test_unknown_voice_lists_the_presets(provider):
    with pytest.raises(ValueError) as exc:
        await provider.synthesize("Hi.", voice="af_heart")
    message = str(exc.value)
    assert "af_heart" in message
    for voice in ("vox_clara", "vox_arthur", "vox_lily", "vox_daniel"):
        assert voice in message
    assert not provider._model.calls


@pytest.mark.asyncio
async def test_speed_and_language_are_ignored(provider):
    await provider.synthesize("Hi.", voice="vox_clara", speed=0.5, language="English")
    call = provider._model.calls[-1]
    assert not {"speed", "language"} & call.keys()


@pytest.mark.asyncio
async def test_mp3_by_default_and_wav_on_request(provider):
    mp3 = await provider.synthesize("Hi.")
    assert mp3[:3] == b"ID3" or mp3[0] == 0xFF
    assert sf.info(io.BytesIO(mp3)).samplerate == _SR

    wav = await provider.synthesize("Hi.", output_format="wav")
    assert wav[:4] == b"RIFF"
    assert sf.info(io.BytesIO(wav)).samplerate == _SR


@pytest.mark.asyncio
async def test_output_is_normalized_to_the_configured_loudness(provider):
    loudness, peak_db = _lufs(await provider.synthesize("Hi.", output_format="wav"))
    assert loudness == pytest.approx(-25.0, abs=0.3)
    assert peak_db <= -1.0 + 0.01


def test_peak_ceiling_caps_the_gain():
    wav = _tone(amplitude=0.01)
    wav[_SR // 2] = 0.9
    out = normalize_loudness(wav, _SR, target_lufs=-10.0, peak_dbfs=-1.0)
    assert 20 * np.log10(np.max(np.abs(out))) == pytest.approx(-1.0, abs=0.01)


def test_silence_and_short_clips_do_not_fail():
    assert not normalize_loudness(np.zeros(0, dtype=np.float32), _SR, -25.0, -1.0).size
    silent = normalize_loudness(np.zeros(_SR, dtype=np.float32), _SR, -25.0, -1.0)
    assert not silent.any()
    short = normalize_loudness(_tone(seconds=0.1), _SR, -25.0, -1.0)
    assert short.size == int(_SR * 0.1)


@pytest.mark.asyncio
async def test_uploaded_reference_with_transcript_uses_continuation(provider):
    ref = io.BytesIO()
    sf.write(ref, _tone(), _SR, format="WAV")
    await provider.synthesize(
        "Hello.", reference_audio=ref.getvalue(), reference_filename="me.wav",
        reference_text="This is me.", voice="ignored",
    )
    call = provider._model.calls[-1]
    assert call["_reference_existed"]
    assert call["prompt_wav_path"] == call["reference_wav_path"]
    assert call["prompt_wav_path"].endswith(".wav")
    assert call["prompt_text"] == "This is me. "
    assert _removed(call["reference_wav_path"])


@pytest.mark.asyncio
async def test_uploaded_reference_without_transcript_clones_timbre_only(provider):
    await provider.synthesize("Hello.", reference_audio=b"RIFF" + b"\x00" * 40)
    call = provider._model.calls[-1]
    assert call["_reference_existed"]
    assert "prompt_wav_path" not in call
    assert "prompt_text" not in call
