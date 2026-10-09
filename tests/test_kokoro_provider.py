"""Kokoro picks the G2P pipeline by the voice's language and shares one model between them."""
from __future__ import annotations

import sys
import types

import pytest

from app.config import ModelConfig
from app.providers.tts.kokoro import KokoroTtsProvider, _lang_code


class _FakePipeline:
    def __init__(self, lang_code, repo_id=None, model=True, device=None):
        self.lang_code = lang_code
        self.repo_id = repo_id
        self.device = device
        self.model = object() if model is True else model


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KPipeline=_FakePipeline))
    config = ModelConfig(
        id="kokoro-82m",
        display_name="Kokoro",
        category="tts",
        provider_class="KokoroTtsProvider",
        model={"hub_id": "hexgrad/Kokoro-82M", "device": "cpu", "default_params": {"voice": "bf_emma"}},
    )
    return KokoroTtsProvider(config)


@pytest.mark.parametrize(
    ("voice", "code"),
    [("af_heart", "a"), ("bm_george", "b"), ("ef_dora", "e"), ("zf_xiaobei", "z"), ("custom", "a")],
)
def test_lang_code_comes_from_the_voice(voice, code):
    assert _lang_code(voice, "a") == code


@pytest.mark.asyncio
async def test_load_builds_the_default_voice_pipeline(provider):
    await provider.load("models")
    pipeline = await provider._pipeline_for("b")
    assert pipeline.repo_id == "hexgrad/Kokoro-82M"
    assert pipeline.device == "cpu"


@pytest.mark.asyncio
async def test_other_languages_share_the_model(provider):
    await provider.load("models")
    british = await provider._pipeline_for("b")
    american = await provider._pipeline_for("a")
    assert american.lang_code == "a"
    assert american.model is british.model
    assert await provider._pipeline_for("a") is american
