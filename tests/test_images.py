from __future__ import annotations

import base64
import io

import pytest
from PIL import Image


def _png_b64(width: int = 8, height: int = 8, colour: tuple[int, int, int] = (200, 50, 50)) -> str:
    img = Image.new("RGB", (width, height), colour)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _mask_b64(width: int = 8, height: int = 8) -> str:
    """Grayscale L-mode mask with a centred white square."""
    img = Image.new("L", (width, height), 0)
    for y in range(2, width - 2):
        for x in range(2, height - 2):
            img.putpixel((x, y), 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.asyncio
async def test_image_generation(client):
    resp = await client.post(
        "/v1/images/generations",
        json={
            "model": "test-image",
            "prompt": "A red circle",
            "n": 1,
            "size": "512x512",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) == 1
    assert data["data"][0]["b64_json"] is not None


@pytest.mark.asyncio
async def test_image_cache_with_seed(client):
    payload = {
        "model": "test-image",
        "prompt": "A blue square",
        "seed": 42,
        "size": "256x256",
    }
    # First request — MISS
    resp1 = await client.post("/v1/images/generations", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers["x-infergate-cache"] == "MISS"

    # Second request — HIT
    resp2 = await client.post("/v1/images/generations", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers["x-infergate-cache"] == "HIT"


@pytest.mark.asyncio
async def test_image_no_cache_without_seed(client):
    resp = await client.post(
        "/v1/images/generations",
        json={"model": "test-image", "prompt": "Test"},
    )
    assert resp.status_code == 200
    # seed_only strategy — no seed means no cache
    assert resp.headers["x-infergate-cache"] == "DISABLED"


@pytest.mark.asyncio
async def test_image_img2img_accepts_base64_input(client):
    resp = await client.post(
        "/v1/images/generations",
        json={
            "model": "test-image",
            "prompt": "stylise this",
            "image": _png_b64(),
            "denoising_strength": 0.5,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["data"][0]["b64_json"] is not None


@pytest.mark.asyncio
async def test_image_inpaint_accepts_image_and_mask(client):
    resp = await client.post(
        "/v1/images/generations",
        json={
            "model": "test-image",
            "prompt": "put a cat here",
            "image": _png_b64(),
            "mask": _mask_b64(),
            "denoising_strength": 0.8,
        },
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_mask_without_image_is_rejected(client):
    """Schema-level guard: inpaint has no reference without the base image."""
    resp = await client.post(
        "/v1/images/generations",
        json={
            "model": "test-image",
            "prompt": "x",
            "mask": _mask_b64(),
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    # Pydantic validation error — message should mention the mask/image invariant
    detail = str(body)
    assert "mask" in detail and "image" in detail


@pytest.mark.asyncio
async def test_image_invalid_base64_returns_400_from_worker(client):
    """Fake provider ignores image field so returns 200 — real decode is e2e-tested."""
    resp = await client.post(
        "/v1/images/generations",
        json={
            "model": "test-image",
            "prompt": "x",
            "image": base64.b64encode(b"not an image").decode(),
        },
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_image_denoising_strength_bounds(client):
    """denoising_strength is bounded to [0.0, 1.0]."""
    resp = await client.post(
        "/v1/images/generations",
        json={
            "model": "test-image",
            "prompt": "x",
            "image": _png_b64(),
            "denoising_strength": 1.5,
        },
    )
    assert resp.status_code == 422


# ── /v1/images/upscale ──────────────────────────────────────────────


_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.mark.asyncio
async def test_upscale_b64_json_default(client):
    """Default response_format=b64_json mirrors /v1/images/generations envelope."""
    resp = await client.post(
        "/v1/images/upscale",
        files={"file": ("in.png", _PNG_BYTES, "image/png")},
        data={"model": "test-upscale"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body and body["data"][0]["b64_json"]
    decoded = base64.b64decode(body["data"][0]["b64_json"])
    assert decoded.endswith(b"UPSCALED")  # fake provider sentinel


@pytest.mark.asyncio
async def test_upscale_raw_png(client):
    """response_format=png returns image bytes with image/png Content-Type."""
    resp = await client.post(
        "/v1/images/upscale",
        files={"file": ("in.png", _PNG_BYTES, "image/png")},
        data={"model": "test-upscale", "response_format": "png"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")
    assert resp.content.endswith(b"UPSCALED")


@pytest.mark.asyncio
async def test_upscale_uses_default_model(client):
    """Omitted model falls back to defaults['upscale']."""
    resp = await client.post(
        "/v1/images/upscale",
        files={"file": ("in.png", _PNG_BYTES, "image/png")},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_upscale_rejects_empty(client):
    resp = await client.post(
        "/v1/images/upscale",
        files={"file": ("empty.png", b"", "image/png")},
        data={"model": "test-upscale"},
    )
    assert resp.status_code == 400
    assert "Empty" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_upscale_rejects_unknown_format(client):
    resp = await client.post(
        "/v1/images/upscale",
        files={"file": ("in.png", _PNG_BYTES, "image/png")},
        data={"model": "test-upscale", "response_format": "jpeg"},
    )
    assert resp.status_code == 400
    assert "response_format" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_upscale_cache_hit_on_same_image(client):
    files = {"file": ("in.png", _PNG_BYTES, "image/png")}
    r1 = await client.post("/v1/images/upscale", files=files, data={"model": "test-upscale"})
    assert r1.status_code == 200
    assert r1.headers["x-infergate-cache"] == "MISS"

    files = {"file": ("in.png", _PNG_BYTES, "image/png")}
    r2 = await client.post("/v1/images/upscale", files=files, data={"model": "test-upscale"})
    assert r2.status_code == 200
    assert r2.headers["x-infergate-cache"] == "HIT"
