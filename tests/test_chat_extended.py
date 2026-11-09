"""Extended chat tests: parameters, cache, edge cases."""
from __future__ import annotations

import pytest

from app.routers.chat import chat_params
from app.schemas.chat import ChatCompletionRequest


async def test_chat_with_temperature(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.5,
    })
    assert resp.status_code == 200


async def test_chat_with_top_p(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "top_p": 0.8,
    })
    assert resp.status_code == 200


async def test_chat_with_max_tokens(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
    })
    assert resp.status_code == 200


async def test_chat_with_response_format(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    })
    assert resp.status_code == 200


async def test_chat_with_thinking(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "thinking": False,
    })
    assert resp.status_code == 200


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


async def test_chat_skip_cache(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "test-text", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"X-InferGate-No-Cache": "true"},
    )
    assert resp.status_code == 200
    assert resp.headers["x-infergate-cache"] == "SKIP"


async def test_chat_validation_temperature_too_high(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 5.0,
    })
    assert resp.status_code == 422


async def test_chat_validation_empty_messages(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [],
    })
    assert resp.status_code == 422


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


async def test_image_validation_n_too_high(client):
    resp = await client.post("/v1/images/generations", json={
        "model": "test-image",
        "prompt": "test",
        "n": 999,
    })
    assert resp.status_code == 422


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


async def test_audio_validation_speed_too_high(client):
    resp = await client.post("/v1/audio/speech", json={
        "model": "test-tts",
        "input": "hello",
        "speed": 10.0,
    })
    assert resp.status_code == 422


def _params(**fields):
    return chat_params(ChatCompletionRequest(messages=[{"role": "user", "content": "Hi"}], **fields))


@pytest.mark.parametrize(("fields", "params"), [
    ({"reasoning_effort": "none"}, {"thinking": False}),
    ({"reasoning_effort": "low"}, {"thinking": True, "reasoning_effort": "low"}),
    ({"reasoning_effort": "high", "thinking": False}, {"thinking": False}),
    ({"thinking": True}, {"thinking": True}),
])
def test_reasoning_effort_drives_thinking_unless_thinking_is_given(fields, params):
    assert _params(**fields) == params


def test_json_schema_reaches_the_provider():
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}

    params = _params(response_format={
        "type": "json_schema", "json_schema": {"name": "note", "strict": True, "schema": schema},
    })

    assert params == {"response_format": "json_schema", "json_schema": schema}


async def test_chat_accepts_openai_structured_output_and_reasoning_effort(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "none",
        "response_format": {"type": "json_schema", "json_schema": {"name": "note", "schema": {"type": "object"}}},
    })
    assert resp.status_code == 200


@pytest.mark.parametrize("response_format", [
    {"type": "json_schema"},
    {"type": "json_object", "json_schema": {"name": "note"}},
])
async def test_json_schema_goes_only_with_its_type(client, response_format):
    resp = await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": response_format,
    })
    assert resp.status_code == 422
