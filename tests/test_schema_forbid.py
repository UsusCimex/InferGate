"""extra='forbid' on public request schemas: typos and unsupported fields
must surface as a 422 with a field-naming message, not be silently dropped."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_chat_rejects_unknown_root_field(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-text",
            "messages": [{"role": "user", "content": "Hi"}],
            "temprature": 0.5,  # typo
        },
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["type"] == "invalid_request"
    assert err["param"] == "temprature"
    assert "unknown field" in err["message"]


@pytest.mark.asyncio
async def test_chat_rejects_unknown_nested_message_field(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-text",
            "messages": [{"role": "user", "content": "Hi", "tool_calls": []}],
        },
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["type"] == "invalid_request"
    assert err["param"] == "messages.0.tool_calls"


@pytest.mark.asyncio
async def test_chat_rejects_unknown_response_format_field(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-text",
            "messages": [{"role": "user", "content": "Hi"}],
            "response_format": {"type": "json_schema", "json_schema": {}},
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["param"] == "response_format.json_schema"


@pytest.mark.asyncio
async def test_audio_rejects_language_typo(client):
    # The TODO's motivating example: `langauge` (transposed letters).
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "test-tts",
            "input": "hello",
            "langauge": "en",
        },
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["param"] == "langauge"
    assert "unknown field" in err["message"]


@pytest.mark.asyncio
async def test_images_rejects_unknown_root_field(client):
    resp = await client.post(
        "/v1/images/generations",
        json={
            "model": "test-image",
            "prompt": "a cat",
            "wdith": "512x512",  # typo for width-like param
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["param"] == "wdith"


@pytest.mark.asyncio
async def test_images_rejects_unknown_lora_field(client):
    resp = await client.post(
        "/v1/images/generations",
        json={
            "model": "test-image",
            "prompt": "a cat",
            "loras": [{"id": "user/repo", "wieght": 0.5}],  # typo
        },
    )
    assert resp.status_code == 422
    assert "loras.0.wieght" in resp.json()["error"]["param"]


@pytest.mark.asyncio
async def test_known_fields_still_work(client):
    # Regression guard: extra='forbid' must not affect legitimate requests.
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-text",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "response_format": {"type": "text"},
        },
    )
    assert resp.status_code == 200
