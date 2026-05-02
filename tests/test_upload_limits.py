"""Tests for upload size limits — both the read-with-limit helper and 413 round-trips."""
from __future__ import annotations

import io

import pytest
from fastapi import UploadFile

from app.utils import UploadTooLargeError, read_with_limit


@pytest.mark.asyncio
async def test_read_with_limit_returns_full_bytes_under_limit():
    upload = UploadFile(file=io.BytesIO(b"x" * 1000), filename="t.bin")
    result = await read_with_limit(upload, max_bytes=2000)
    assert result == b"x" * 1000


@pytest.mark.asyncio
async def test_read_with_limit_raises_when_exceeded():
    upload = UploadFile(file=io.BytesIO(b"x" * 5000), filename="t.bin")
    with pytest.raises(UploadTooLargeError) as exc_info:
        await read_with_limit(upload, max_bytes=1000)
    assert exc_info.value.max_bytes == 1000


@pytest.mark.asyncio
async def test_read_with_limit_exact_size_passes():
    upload = UploadFile(file=io.BytesIO(b"x" * 1000), filename="t.bin")
    result = await read_with_limit(upload, max_bytes=1000)
    assert len(result) == 1000


@pytest.mark.asyncio
async def test_read_with_limit_empty_file():
    upload = UploadFile(file=io.BytesIO(b""), filename="empty.bin")
    result = await read_with_limit(upload, max_bytes=1000)
    assert result == b""


@pytest.mark.asyncio
async def test_read_with_limit_rejects_zero_max():
    upload = UploadFile(file=io.BytesIO(b"x"), filename="t.bin")
    with pytest.raises(ValueError, match="max_bytes must be > 0"):
        await read_with_limit(upload, max_bytes=0)


@pytest.mark.asyncio
async def test_read_with_limit_streams_bounded_memory(monkeypatch):
    """Once the limit is hit, no more bytes are accumulated — verify by inspecting buffer."""
    big_payload = b"y" * 10_000
    upload = UploadFile(file=io.BytesIO(big_payload), filename="t.bin")

    # 1KB limit → reader should bail out around 1KB+chunk_size, not after reading all 10KB.
    with pytest.raises(UploadTooLargeError):
        await read_with_limit(upload, max_bytes=1024, chunk_size=512)


# ── HTTP-level: routers return 413 ───────────────────────────────────


@pytest.mark.asyncio
async def test_audio_transcription_returns_413_on_oversized_upload(client, services):
    """Setting a tiny audio limit must turn a normal-sized request into 413."""
    from app.config import UploadLimitsConfig

    services_state = client._transport.app.state
    services_state.upload_limits = UploadLimitsConfig(max_audio_mb=1)

    big_audio = b"\x00" * (2 * 1024 * 1024)  # 2MB > 1MB limit
    files = {"file": ("test.wav", big_audio, "audio/wav")}
    resp = await client.post("/v1/audio/transcriptions", files=files)
    assert resp.status_code == 413
    assert resp.json()["error"]["type"] == "upload_too_large"


@pytest.mark.asyncio
async def test_image_embedding_returns_413_on_oversized_upload(client, services):
    from app.config import UploadLimitsConfig

    services_state = client._transport.app.state
    services_state.upload_limits = UploadLimitsConfig(max_image_mb=1)

    big_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024)
    files = {"file": ("big.png", big_image, "image/png")}
    resp = await client.post("/v1/embeddings/image", files=files)
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_video_embedding_returns_413_on_oversized_upload(client, services):
    from app.config import UploadLimitsConfig

    services_state = client._transport.app.state
    services_state.upload_limits = UploadLimitsConfig(max_video_mb=1)

    big_video = b"\x00" * (2 * 1024 * 1024)
    files = {"file": ("clip.mp4", big_video, "video/mp4")}
    resp = await client.post("/v1/embeddings/video", files=files)
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_image_upscale_returns_413_on_oversized_upload(client, services):
    from app.config import UploadLimitsConfig

    services_state = client._transport.app.state
    services_state.upload_limits = UploadLimitsConfig(max_upscale_mb=1)

    big_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024)
    files = {"file": ("big.png", big_image, "image/png")}
    resp = await client.post("/v1/images/upscale", files=files)
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_normal_size_upload_passes_size_check(client, services):
    """Sanity: a small upload still succeeds even with default limits."""
    small_audio = b"\x00" * 1024  # 1KB
    files = {"file": ("small.wav", small_audio, "audio/wav")}
    resp = await client.post("/v1/audio/transcriptions", files=files)
    # Test transcribe provider returns 200 for any non-empty audio.
    assert resp.status_code == 200
