"""llama.cpp text provider: GGUF lookup in the HF cache, the chat body it sends and the answers it relays."""
from __future__ import annotations

import json

import httpx
import pytest

from app.config import ModelConfig
from app.providers.text import llamacpp_provider
from app.providers.text.llamacpp_provider import LlamaCppTextProvider

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "gemma-4-12b-it-qat-q4_0.gguf",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"ok\": true}"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
}


@pytest.fixture
def provider():
    config = ModelConfig(
        id="gemma-4-12b",
        display_name="Gemma 4 12B",
        category="text",
        provider_class="LlamaCppTextProvider",
        model={
            "hub_id": "google/gemma-4-12B-it-qat-q4_0-gguf",
            "gguf_file": "gemma-4-12b-it-qat-q4_0.gguf",
            "default_params": {"max_tokens": 4096, "temperature": 0.7},
        },
    )
    return LlamaCppTextProvider(config)


def answer_with(provider, handler):
    provider._client = httpx.AsyncClient(base_url="http://llama", transport=httpx.MockTransport(handler))


def test_the_gguf_comes_from_the_snapshot_main_points_to(provider, tmp_path):
    cache = tmp_path / "models--google--gemma-4-12B-it-qat-q4_0-gguf"
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text("abc123\n")
    gguf = cache / "snapshots" / "abc123" / "gemma-4-12b-it-qat-q4_0.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")

    assert provider.gguf_path(str(tmp_path)) == str(gguf)


def test_the_body_asks_for_json_and_leaves_thinking_to_the_server(provider):
    body = provider.request_body(
        [{"role": "user", "content": "hi"}], {"response_format": "json_object", "thinking": True, "temperature": 0.2},
        stream=False,
    )

    assert body == {
        "max_tokens": 4096,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "model": "gemma-4-12b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


async def test_an_answer_carries_the_model_id(provider):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=COMPLETION)

    answer_with(provider, handler)

    result = await provider.generate([{"role": "user", "content": "hi"}], max_tokens=50)

    assert result["model"] == "gemma-4-12b"
    assert result["choices"][0]["message"]["content"] == "{\"ok\": true}"
    assert sent[0]["max_tokens"] == 50


async def test_a_request_the_server_rejects_is_a_client_error(provider):
    answer_with(provider, lambda request: httpx.Response(400, json={"error": {"message": "bad grammar"}}))

    with pytest.raises(ValueError, match="bad grammar"):
        await provider.generate([{"role": "user", "content": "hi"}])


async def test_a_stream_relays_the_server_events(provider):
    events = 'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\ndata: [DONE]\n\n'
    answer_with(provider, lambda request: httpx.Response(200, text=events))

    chunks = [chunk async for chunk in provider.generate_stream([{"role": "user", "content": "hi"}])]

    assert chunks == ['data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n', "data: [DONE]\n\n"]


async def test_a_server_that_exits_while_loading_fails_the_load(provider, tmp_path, monkeypatch):
    gguf = tmp_path / "models--google--gemma-4-12B-it-qat-q4_0-gguf" / "snapshots" / "main" / "gemma-4-12b-it-qat-q4_0.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")

    class ExitedProcess:
        returncode = 1

    async def start(*args, **kwargs):
        return ExitedProcess()

    monkeypatch.setattr(llamacpp_provider.asyncio, "create_subprocess_exec", start)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        await provider.load(str(tmp_path))
    assert not provider.is_loaded()
