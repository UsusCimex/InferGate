"""Models that declare capabilities.voices reject other voices at the API boundary."""
from __future__ import annotations

import pytest

from app.config import (
    ModelCacheConfig,
    ModelCapabilities,
    ModelConfig,
    ModelMetadata,
    ModelQueueConfig,
)
from tests.conftest import FakeTtsProvider

_VOICES = ["vox_clara", "vox_daniel"]


@pytest.fixture
def preset_model(services):
    cfg = ModelConfig(
        id="test-preset-tts",
        display_name="test-preset-tts",
        category="tts",
        provider_class="FakeTtsProvider",
        enabled=True,
        model={"hub_id": "test/preset", "vram_mb": 1000},
        cache=ModelCacheConfig(enabled=False, strategy="never", max_size_mb=0),
        queue=ModelQueueConfig(priority="medium", timeout_seconds=30, max_concurrent=1),
        metadata=ModelMetadata(),
        capabilities=ModelCapabilities(voices=_VOICES),
    )
    provider = FakeTtsProvider(cfg)
    services["manager"]._registry["test-preset-tts"] = provider
    services["scheduler"].register_model("test-preset-tts", 1)
    yield provider


@pytest.mark.asyncio
async def test_unknown_voice_is_rejected_before_loading(client, preset_model):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "test-preset-tts", "input": "Hello", "voice": "af_heart"},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request"
    assert err["param"] == "voice"
    assert err["voices"] == _VOICES
    assert "vox_clara, vox_daniel" in err["message"]
    assert not preset_model.is_loaded()


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", ["vox_daniel", "default"])
async def test_preset_and_default_voice_are_accepted(client, preset_model, voice):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "test-preset-tts", "input": "Hello", "voice": voice},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_model_without_voice_list_accepts_any_voice(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "test-tts", "input": "Hello", "voice": "anything"},
    )
    assert resp.status_code == 200
