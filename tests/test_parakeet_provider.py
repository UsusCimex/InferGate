"""Parakeet: the export files it fetches, the samples it hands to onnx-asr and the answer it builds."""
from __future__ import annotations

import io
import sys
import types
import wave

import numpy as np
import pytest

from app.config import ModelConfig
from app.providers.stt import parakeet_provider
from app.providers.stt.parakeet_provider import ParakeetProvider


class _FakeAsr:
    def __init__(self):
        self.calls = []

    def recognize(self, waveform, sample_rate=16_000):
        self.calls.append((waveform, sample_rate))
        return " the cat sat on the mat "


@pytest.fixture
def provider():
    config = ModelConfig(
        id="parakeet-tdt-0.6b-v3",
        display_name="Parakeet TDT 0.6B v3",
        category="stt",
        provider_class="ParakeetProvider",
        model={"hub_id": "istupakov/parakeet-tdt-0.6b-v3-onnx", "asr_model": "nemo-parakeet-tdt-0.6b-v3",
               "quantization": "int8", "cpu_threads": 2},
    )
    return ParakeetProvider(config)


@pytest.fixture
def loaded(provider, monkeypatch):
    seen = {}
    asr = _FakeAsr()

    def snapshot_download(repo_id, cache_dir, allow_patterns):
        seen.update(repo_id=repo_id, cache_dir=cache_dir, allow_patterns=allow_patterns)
        return "/models/snapshot"

    def load_model(model, path, quantization, sess_options, providers):
        seen.update(model=model, path=path, quantization=quantization, threads=sess_options.intra_op_num_threads,
                    providers=providers)
        return asr

    class _SessionOptions:
        intra_op_num_threads = 0

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(sys.modules, "onnx_asr", types.SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(SessionOptions=_SessionOptions))
    return provider, asr, seen


def _wav(seconds: float, rate: int = 8_000, channels: int = 2) -> bytes:
    frames = np.zeros(int(seconds * rate) * channels, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames.tobytes())
    return buf.getvalue()


def test_only_the_quantized_export_is_fetched(provider):
    assert provider.allow_patterns() == [
        "config.json", "vocab.txt", "encoder-model.int8.onnx*", "decoder_joint-model.int8.onnx*",
    ]


def test_the_full_precision_export_takes_its_external_weights(provider):
    provider.config.model["quantization"] = ""
    assert "encoder-model.onnx*" in provider.allow_patterns()


@pytest.mark.asyncio
async def test_load_opens_the_snapshot_on_the_cpu(loaded):
    provider, _, seen = loaded
    await provider.load("/app/models")

    assert seen == {
        "repo_id": "istupakov/parakeet-tdt-0.6b-v3-onnx", "cache_dir": "/app/models",
        "allow_patterns": provider.allow_patterns(), "model": "nemo-parakeet-tdt-0.6b-v3",
        "path": "/models/snapshot", "quantization": "int8", "threads": 2, "providers": ["CPUExecutionProvider"],
    }
    assert provider.is_loaded()


@pytest.mark.asyncio
async def test_transcribe_hands_16_khz_samples_and_trims_the_text(loaded, monkeypatch):
    provider, asr, _ = loaded
    monkeypatch.setattr(parakeet_provider, "decode_audio", lambda audio: np.zeros(32_000, dtype=np.float32))
    await provider.load("/app/models")

    result = await provider.transcribe(b"audio", response_format="verbose_json", language="en")

    assert result == {"text": "the cat sat on the mat", "language": "en", "duration": 2.0,
                      "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "the cat sat on the mat"}]}
    assert asr.calls[0][1] == 16_000


@pytest.mark.asyncio
async def test_silence_without_samples_gives_empty_text(loaded, monkeypatch):
    provider, asr, _ = loaded
    monkeypatch.setattr(parakeet_provider, "decode_audio", lambda audio: np.zeros(0, dtype=np.float32))
    await provider.load("/app/models")

    assert await provider.transcribe(b"audio") == {"text": ""}
    assert asr.calls == []


def test_decode_resamples_to_16_khz_mono():
    pytest.importorskip("av")
    samples = parakeet_provider.decode_audio(_wav(1.5))

    assert samples.dtype == np.float32
    assert abs(len(samples) - 24_000) < 400
