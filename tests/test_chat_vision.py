from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from app.config import ModelCapabilities, UploadLimitsConfig
from app.providers.text._chat_images import decode_data_url, image_urls, with_system_instruction


def _data_url(size: tuple[int, int] = (8, 6)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _request(url: str) -> dict:
    return {
        "model": "test-text",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "What is drawn?"},
        ]}],
    }


@pytest.fixture
def vision_model(services):
    config = services["manager"].get_config("test-text")
    config.capabilities = ModelCapabilities(vision=True)
    return services["manager"].get("test-text")


@pytest.mark.asyncio
async def test_image_rejected_for_model_without_vision(client):
    resp = await client.post("/v1/chat/completions", json=_request(_data_url()))
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "vision_not_supported"


@pytest.mark.asyncio
async def test_image_parts_reach_provider_without_null_fields(client, vision_model, monkeypatch):
    seen = []
    original = vision_model.generate

    async def recording(messages, **params):
        seen.append(messages)
        return await original(messages, **params)

    monkeypatch.setattr(vision_model, "generate", recording)
    url = _data_url()
    resp = await client.post("/v1/chat/completions", json=_request(url))
    assert resp.status_code == 200
    assert seen[0][0]["content"] == [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "What is drawn?"},
    ]


@pytest.mark.asyncio
async def test_remote_image_url_rejected(client, vision_model):
    resp = await client.post("/v1/chat/completions", json=_request("https://example.com/cat.png"))
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_image"


@pytest.mark.asyncio
async def test_oversized_image_returns_413(client, vision_model):
    client._transport.app.state.upload_limits = UploadLimitsConfig(max_image_mb=1)
    url = "data:image/png;base64," + "A" * (2 * 1024 * 1024)
    resp = await client.post("/v1/chat/completions", json=_request(url))
    assert resp.status_code == 413


@pytest.mark.asyncio
@pytest.mark.parametrize("part", [
    {"type": "audio", "audio": "x"},
    {"type": "text", "text": "hi", "extra": 1},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "extra": 1}},
])
async def test_malformed_content_part_is_422(client, part):
    body = {"model": "test-text", "messages": [{"role": "user", "content": [part]}]}
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_string_content_still_accepted(client):
    body = {"model": "test-text", "messages": [{"role": "user", "content": "Hi"}]}
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200


def test_decode_data_url_returns_rgb_image():
    image = decode_data_url(_data_url((8, 6)))
    assert image.mode == "RGB"
    assert image.size == (8, 6)


@pytest.mark.parametrize("url", [
    "https://example.com/cat.png",
    "file:///etc/passwd",
    "data:image/png;base64,@@@",
    "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
])
def test_decode_data_url_rejects_non_images(url):
    with pytest.raises(ValueError):
        decode_data_url(url)


def test_image_urls_in_message_order():
    first, second = _data_url((2, 2)), _data_url((3, 3))
    messages = [
        {"role": "system", "content": "Judge pictures."},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": first}}]},
        {"role": "user", "content": [
            {"type": "text", "text": "and this"},
            {"type": "image_url", "image_url": {"url": second}},
        ]},
    ]
    assert image_urls(messages) == [first, second]


def test_system_instruction_flattens_list_content():
    messages = [{"role": "system", "content": [{"type": "text", "text": "Judge pictures."}]},
                {"role": "user", "content": "Hi"}]
    result = with_system_instruction(messages, "Respond with valid JSON only.")
    assert result[0]["content"] == "Judge pictures.\nRespond with valid JSON only."
    assert messages[0]["content"] == [{"type": "text", "text": "Judge pictures."}]


def test_system_instruction_prepended_without_system_message():
    result = with_system_instruction([{"role": "user", "content": "Hi"}], "Respond with valid JSON only.")
    assert result[0] == {"role": "system", "content": "Respond with valid JSON only."}
