"""vLLM answers: json_object holds the reply to the JSON grammar, and a cut-off reply says so."""
from __future__ import annotations

import json
import sys
import types

import pytest

from app.config import ModelConfig
from app.providers.text.vllm_provider import VllmTextProvider


class _Engine:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason
        self.sampling = None

    async def generate(self, prompt, sampling, request_id):
        self.sampling = sampling
        out = types.SimpleNamespace(text='{"a": 1}', token_ids=[1, 2, 3], finish_reason=self.finish_reason)
        yield types.SimpleNamespace(outputs=[out], prompt_token_ids=[7, 8])


@pytest.fixture
def vllm(monkeypatch):
    sampling_params = types.SimpleNamespace(StructuredOutputsParams=lambda **kwargs: kwargs)
    module = types.SimpleNamespace(SamplingParams=lambda **kwargs: kwargs, sampling_params=sampling_params)
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)


def _provider(finish_reason: str | None = "stop") -> VllmTextProvider:
    provider = VllmTextProvider(ModelConfig(
        id="qwen3.5-4b", display_name="Qwen", category="text", provider_class="VllmTextProvider",
        model={"hub_id": "cyankiwi/Qwen3.5-4B-AWQ-4bit"},
    ))
    provider._engine = _Engine(finish_reason)
    provider._build_prompt = lambda messages, thinking=True, images=None: "prompt"
    return provider


MESSAGES = [{"role": "user", "content": "A tale as JSON."}]


@pytest.mark.asyncio
async def test_json_object_holds_the_answer_to_the_json_grammar(vllm):
    provider = _provider()

    await provider.generate(MESSAGES, response_format="json_object", thinking=False)

    assert provider._engine.sampling["structured_outputs"] == {"json_object": True}


@pytest.mark.asyncio
async def test_thinking_leaves_the_think_block_free_of_the_grammar(vllm):
    provider = _provider()

    await provider.generate(MESSAGES, response_format="json_object", thinking=True)

    assert provider._engine.sampling["structured_outputs"] is None


@pytest.mark.asyncio
async def test_plain_text_has_no_grammar(vllm):
    provider = _provider()

    await provider.generate(MESSAGES, thinking=False)

    assert provider._engine.sampling["structured_outputs"] is None


@pytest.mark.asyncio
async def test_an_answer_cut_at_max_tokens_finishes_with_length(vllm):
    result = await _provider("length").generate(MESSAGES, thinking=False)

    assert result["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_the_stream_ends_with_the_engine_finish_reason(vllm):
    provider = _provider("length")

    chunks = [c async for c in provider.generate_stream(MESSAGES, response_format="json_object", thinking=False)]

    final = json.loads(chunks[-2].removeprefix("data: "))
    assert final["choices"][0]["finish_reason"] == "length"
    assert provider._engine.sampling["structured_outputs"] == {"json_object": True}
