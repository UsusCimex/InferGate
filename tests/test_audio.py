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


@pytest.mark.asyncio
async def test_tts_language_routed_into_cache_key(client):
    """`language` must reach the cache layer: two requests identical except
    for language should both be MISS (distinct cache keys), not HIT."""
    base = {"model": "test-tts", "input": "Hola"}

    r1 = await client.post("/v1/audio/speech", json={**base, "language": "Spanish"})
    assert r1.status_code == 200
    assert r1.headers["x-infergate-cache"] == "MISS"

    r2 = await client.post("/v1/audio/speech", json={**base, "language": "French"})
    assert r2.status_code == 200
    assert r2.headers["x-infergate-cache"] == "MISS"


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
        data={"model": "test-stt", "response_format": "docx"},
    )
    assert resp.status_code == 400
    assert "response_format" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_stt_srt_format(client):
    """response_format=srt renders segments as SubRip with comma-millisecond timestamps."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _FAKE_AUDIO, "audio/wav")},
        data={"model": "test-stt", "response_format": "srt"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-subrip")
    # Fake provider returns a single segment: 0.0 → 1.0 "hello world"
    body = resp.text
    assert "1\n" in body
    assert "00:00:00,000 --> 00:00:01,000" in body
    assert "hello world" in body


@pytest.mark.asyncio
async def test_stt_vtt_format(client):
    """response_format=vtt renders WEBVTT with dot-millisecond timestamps."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _FAKE_AUDIO, "audio/wav")},
        data={"model": "test-stt", "response_format": "vtt"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/vtt")
    body = resp.text
    assert body.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.000" in body
    assert "hello world" in body


def test_srt_vtt_time_formatting():
    """Unit test for the time-conversion helpers on boundary values."""
    from app.routers.audio import _fmt_srt_time, _fmt_vtt_time
    # Zero, sub-second, round seconds, cross-hour
    assert _fmt_srt_time(0.0) == "00:00:00,000"
    assert _fmt_srt_time(1.5) == "00:00:01,500"
    assert _fmt_srt_time(3725.123) == "01:02:05,123"
    assert _fmt_vtt_time(3725.123) == "01:02:05.123"


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


# ── Voice cloning ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_voice_clone_basic(client):
    """Multipart request with reference_audio returns synthesised audio."""
    resp = await client.post(
        "/v1/audio/speech/voice-clone",
        files={"reference_audio": ("ref.wav", _FAKE_AUDIO, "audio/wav")},
        data={"input": "hello world", "model": "test-tts"},
    )
    assert resp.status_code == 200
    # Fake provider returns a RIFF-prefixed blob regardless of params.
    assert resp.content.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_voice_clone_uses_default_model(client):
    resp = await client.post(
        "/v1/audio/speech/voice-clone",
        files={"reference_audio": ("ref.wav", _FAKE_AUDIO, "audio/wav")},
        data={"input": "hi"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_voice_clone_rejects_empty_reference(client):
    resp = await client.post(
        "/v1/audio/speech/voice-clone",
        files={"reference_audio": ("ref.wav", b"", "audio/wav")},
        data={"input": "hi", "model": "test-tts"},
    )
    assert resp.status_code == 400
    assert "Empty" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_voice_clone_cache_hit_on_identical_request(client):
    """Same input + same reference bytes + same speed → cache HIT."""
    files = {"reference_audio": ("ref.wav", _FAKE_AUDIO, "audio/wav")}
    data = {"input": "same text", "model": "test-tts", "speed": "1.0"}
    r1 = await client.post("/v1/audio/speech/voice-clone", files=files, data=data)
    assert r1.status_code == 200
    assert r1.headers["x-infergate-cache"] == "MISS"

    files = {"reference_audio": ("ref.wav", _FAKE_AUDIO, "audio/wav")}
    r2 = await client.post("/v1/audio/speech/voice-clone", files=files, data=data)
    assert r2.status_code == 200
    assert r2.headers["x-infergate-cache"] == "HIT"


@pytest.mark.asyncio
async def test_voice_clone_different_references_bust_cache(client):
    """Identical text + same model but different reference audio → different cache keys."""
    data = {"input": "same text", "model": "test-tts"}
    r1 = await client.post(
        "/v1/audio/speech/voice-clone",
        files={"reference_audio": ("a.wav", b"AAAA" + _FAKE_AUDIO, "audio/wav")},
        data=data,
    )
    assert r1.headers["x-infergate-cache"] == "MISS"
    r2 = await client.post(
        "/v1/audio/speech/voice-clone",
        files={"reference_audio": ("b.wav", b"BBBB" + _FAKE_AUDIO, "audio/wav")},
        data=data,
    )
    assert r2.headers["x-infergate-cache"] == "MISS"  # different voice → different key
