"""Extended chat tests: parameters, cache, edge cases."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_chat_with_temperature(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.5,
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_with_top_p(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "top_p": 0.8,
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_with_max_tokens(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_with_response_format(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_with_thinking(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "thinking": False,
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_no_model_no_default(client, services):
    """When no model specified and no default, should return 400."""
    old_defaults = services["defaults"].copy()
    services["defaults"].clear()
    try:
        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 400
    finally:
        services["defaults"].update(old_defaults)


@pytest.mark.asyncio
async def test_chat_skip_cache(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "test-text", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"X-InferGate-No-Cache": "true"},
    )
    assert resp.status_code == 200
    assert resp.headers["x-infergate-cache"] == "SKIP"


@pytest.mark.asyncio
async def test_chat_validation_temperature_too_high(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 5.0,
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_validation_empty_messages(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [],
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_image_no_model_no_default(client, services):
    old_defaults = services["defaults"].copy()
    services["defaults"].clear()
    try:
        resp = await client.post("/v1/images/generations", json={
            "prompt": "test",
        })
        assert resp.status_code == 400
    finally:
        services["defaults"].update(old_defaults)


@pytest.mark.asyncio
async def test_image_validation_n_too_high(client):
    resp = await client.post("/v1/images/generations", json={
        "model": "test-image",
        "prompt": "test",
        "n": 999,
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_audio_no_model_no_default(client, services):
    old_defaults = services["defaults"].copy()
    services["defaults"].clear()
    try:
        resp = await client.post("/v1/audio/speech", json={
            "input": "hello",
        })
        assert resp.status_code == 400
    finally:
        services["defaults"].update(old_defaults)


@pytest.mark.asyncio
async def test_audio_validation_speed_too_high(client):
    resp = await client.post("/v1/audio/speech", json={
        "model": "test-tts",
        "input": "hello",
        "speed": 10.0,
    })
    assert resp.status_code == 422
