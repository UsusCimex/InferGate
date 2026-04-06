from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_chat_completion(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-text",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert resp.headers["x-infergate-model"] == "test-text"


@pytest.mark.asyncio
async def test_chat_default_model(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "test-text"


@pytest.mark.asyncio
async def test_chat_unknown_model(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["error"]["message"].lower()
