"""Voice-clone-only TTS models refuse the flat /v1/audio/speech endpoint
with a structured 400 that names /voice-clone as the right alternative —
instead of the 500 that used to leak from the provider's ValueError."""
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


@pytest.fixture
def voice_clone_model(services):
    """Register a voice-clone-only TTS in the existing fixture's manager."""
    cfg = ModelConfig(
        id="test-clone-tts",
        display_name="test-clone-tts",
        category="tts",
        provider_class="FakeTtsProvider",
        enabled=True,
        model={"hub_id": "test/clone", "vram_mb": 1000},
        cache=ModelCacheConfig(enabled=False, strategy="never", max_size_mb=0),
        queue=ModelQueueConfig(priority="medium", timeout_seconds=30, max_concurrent=1),
        metadata=ModelMetadata(),
        capabilities=ModelCapabilities(voice_clone_only=True),
    )
    services["manager"]._registry["test-clone-tts"] = FakeTtsProvider(cfg)
    services["scheduler"].register_model("test-clone-tts", 1)
    yield cfg


@pytest.mark.asyncio
async def test_flat_endpoint_rejects_voice_clone_only_model(client, voice_clone_model):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "test-clone-tts", "input": "Hello"},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "voice_clone_required"
    assert err["endpoint"] == "/v1/audio/speech/voice-clone"
    assert "test-clone-tts" in err["message"]


@pytest.mark.asyncio
async def test_voice_clone_endpoint_still_accepts_voice_clone_only_model(
    client, voice_clone_model
):
    resp = await client.post(
        "/v1/audio/speech/voice-clone",
        data={"model": "test-clone-tts", "input": "Hello"},
        files={"reference_audio": ("ref.wav", b"RIFF" + b"\x00" * 40, "audio/wav")},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_regular_tts_unaffected(client):
    # Regression guard: the default test-tts has voice_clone_only=False,
    # so the fast-path check must not touch it.
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "test-tts", "input": "Hello"},
    )
    assert resp.status_code == 200
