from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_tts(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "test-tts", "input": "Hello world"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_tts_cache(client):
    payload = {"model": "test-tts", "input": "Cache test", "voice": "default"}

    resp1 = await client.post("/v1/audio/speech", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers["x-infergate-cache"] == "MISS"

    resp2 = await client.post("/v1/audio/speech", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers["x-infergate-cache"] == "HIT"


@pytest.mark.asyncio
async def test_tts_skip_cache(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "test-tts", "input": "No cache"},
        headers={"X-InferGate-No-Cache": "true"},
    )
    assert resp.status_code == 200
    assert resp.headers["x-infergate-cache"] == "SKIP"


# ── Transcription ──────────────────────────────────────────────────────

_FAKE_AUDIO = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 40  # pseudo-WAV


@pytest.mark.asyncio
async def test_stt_json(client):
    """Default response_format=json returns strict OpenAI shape `{text: ...}`."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _FAKE_AUDIO, "audio/wav")},
        data={"model": "test-stt"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"text": "hello world"}  # language/duration intentionally stripped


@pytest.mark.asyncio
async def test_stt_text_format(client):
    """response_format=text returns raw text body, not JSON."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _FAKE_AUDIO, "audio/wav")},
        data={"model": "test-stt", "response_format": "text"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "hello world"


@pytest.mark.asyncio
async def test_stt_verbose_json_includes_segments(client):
    """verbose_json passes language/duration/segments through."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _FAKE_AUDIO, "audio/wav")},
        data={"model": "test-stt", "response_format": "verbose_json", "language": "ru"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "hello world"
    assert body["language"] == "ru"  # fake provider echoes the hint
    assert isinstance(body["segments"], list) and body["segments"][0]["text"] == "hello world"


@pytest.mark.asyncio
async def test_stt_uses_default_model(client):
    """Unspecified `model` field picks defaults["stt"]."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _FAKE_AUDIO, "audio/wav")},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_stt_rejects_unknown_format(client):
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _FAKE_AUDIO, "audio/wav")},
        data={"model": "test-stt", "response_format": "srt"},
    )
    assert resp.status_code == 400
    assert "response_format" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_stt_rejects_empty_file(client):
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"model": "test-stt"},
    )
    assert resp.status_code == 400
    assert "Empty" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_stt_cache_hit_on_same_audio(client):
    """Same bytes + same params → cache HIT on second call."""
    files = {"file": ("a.wav", _FAKE_AUDIO, "audio/wav")}
    r1 = await client.post("/v1/audio/transcriptions", files=files, data={"model": "test-stt"})
    assert r1.status_code == 200
    assert r1.headers["x-infergate-cache"] == "MISS"

    files = {"file": ("a.wav", _FAKE_AUDIO, "audio/wav")}  # rebuild — httpx consumes
    r2 = await client.post("/v1/audio/transcriptions", files=files, data={"model": "test-stt"})
    assert r2.status_code == 200
    assert r2.headers["x-infergate-cache"] == "HIT"
    assert r2.json() == {"text": "hello world"}
