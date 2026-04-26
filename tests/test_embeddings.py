from __future__ import annotations

import io
import struct


def _minimal_wav(num_samples: int = 16) -> bytes:
    """Build a minimal valid RIFF/WAVE PCM16 mono 16kHz file."""
    sample_rate = 16000
    bits_per_sample = 16
    num_channels = 1
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
    )
    data_chunk = struct.pack("<4sI", b"data", data_size) + b"\x00" * data_size
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    return struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE") + fmt_chunk + data_chunk


async def test_embeddings_text_string_input(client):
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "test-embed-text", "input": "query: hello"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["model"] == "test-embed-text"
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert item["object"] == "embedding"
    assert item["index"] == 0
    assert isinstance(item["embedding"], list)
    assert len(item["embedding"]) == 8


async def test_embeddings_text_batch_input(client):
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "test-embed-text", "input": ["a", "bb", "ccc"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 3
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    # Different-length strings produce different vectors via the fake provider.
    assert body["data"][0]["embedding"] != body["data"][1]["embedding"]
    assert body["data"][1]["embedding"] != body["data"][2]["embedding"]


async def test_embeddings_text_uses_default_model(client):
    resp = await client.post("/v1/embeddings", json={"input": "no model specified"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "test-embed-text"


async def test_embeddings_text_rejects_extra_field(client):
    resp = await client.post(
        "/v1/embeddings",
        json={"input": "hi", "dimensions": 128},
    )
    assert resp.status_code == 422


async def test_embeddings_audio(client):
    wav = _minimal_wav()
    resp = await client.post(
        "/v1/embeddings/audio",
        files={"file": ("clip.wav", io.BytesIO(wav), "audio/wav")},
        data={"model": "test-embed-audio"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "test-embed-audio"
    assert isinstance(body["embedding"], list)
    assert len(body["embedding"]) == 4


async def test_embeddings_audio_uses_default_model(client):
    wav = _minimal_wav()
    resp = await client.post(
        "/v1/embeddings/audio",
        files={"file": ("clip.wav", io.BytesIO(wav), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "test-embed-audio"


async def test_embeddings_audio_rejects_empty_upload(client):
    resp = await client.post(
        "/v1/embeddings/audio",
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert resp.status_code == 400


_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


async def test_embeddings_image(client):
    resp = await client.post(
        "/v1/embeddings/image",
        files={"file": ("frame.png", io.BytesIO(_MIN_PNG), "image/png")},
        data={"model": "test-embed-multi"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "test-embed-multi"
    assert isinstance(body["embedding"], list)
    assert len(body["embedding"]) == 6


async def test_embeddings_image_uses_default_model(client):
    resp = await client.post(
        "/v1/embeddings/image",
        files={"file": ("frame.png", io.BytesIO(_MIN_PNG), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "test-embed-multi"


async def test_embeddings_image_rejects_empty_upload(client):
    resp = await client.post(
        "/v1/embeddings/image",
        files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert resp.status_code == 400


async def test_embeddings_text_via_multimodal_model(client):
    """SigLIP-style multimodal models can also embed text via /v1/embeddings."""
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "test-embed-multi", "input": "сцена погони"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "test-embed-multi"
    assert len(body["data"]) == 1
    # multimodal fake returns 6-d vectors (same on text + image side)
    assert len(body["data"][0]["embedding"]) == 6


async def test_embeddings_video(client):
    fake_mp4 = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 64
    resp = await client.post(
        "/v1/embeddings/video",
        files={"file": ("clip.mp4", io.BytesIO(fake_mp4), "video/mp4")},
        data={"model": "test-embed-video"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "test-embed-video"
    assert isinstance(body["embedding"], list)
    assert len(body["embedding"]) == 5


async def test_embeddings_video_uses_default_model(client):
    fake_mp4 = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 64
    resp = await client.post(
        "/v1/embeddings/video",
        files={"file": ("clip.mp4", io.BytesIO(fake_mp4), "video/mp4")},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "test-embed-video"


async def test_embeddings_video_rejects_empty_upload(client):
    resp = await client.post(
        "/v1/embeddings/video",
        files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
    )
    assert resp.status_code == 400


async def test_embeddings_text_via_video_model(client):
    """CLIP4Clip-style video models can also embed text via /v1/embeddings."""
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "test-embed-video", "input": "horse chase"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "test-embed-video"
    assert len(body["data"]) == 1
    # video fake returns 5-d vectors (same on text + video side)
    assert len(body["data"][0]["embedding"]) == 5
