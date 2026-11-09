"""vLLM answers: the JSON grammar holds the reply, reasoning comes apart from it, and a cut-off reply says so."""
from __future__ import annotations

import json
import sys
import types

import pytest

from app.config import ModelConfig
from app.providers.text.vllm_provider import VllmTextProvider

END = 9


class _Engine:
    def __init__(self, steps, finish_reason):
        self.steps = steps
        self.finish_reason = finish_reason
        self.sampling = None

    async def generate(self, prompt, sampling, request_id):
        self.sampling = sampling
        for text, token_ids in self.steps:
            out = types.SimpleNamespace(text=text, token_ids=token_ids, finish_reason=self.finish_reason)
            yield types.SimpleNamespace(outputs=[out], prompt_token_ids=[7, 8])


class _Parser:
    """The split of vLLM's Qwen3 parser: everything before </think> is reasoning."""

    def extract_reasoning(self, text, request):
        reasoning, end, content = text.partition("</think>")
        return (reasoning, content or None) if end else (text, None)

    def extract_reasoning_streaming(self, previous, current, delta, previous_ids, current_ids, delta_ids):
        if END in previous_ids:
            return types.SimpleNamespace(reasoning=None, content=delta)
        if END in delta_ids:
            reasoning, _, content = delta.partition("</think>")
            return types.SimpleNamespace(reasoning=reasoning or None, content=content or None)
        return types.SimpleNamespace(reasoning=delta, content=None)


@pytest.fixture
def vllm(monkeypatch):
    sampling_params = types.SimpleNamespace(StructuredOutputsParams=lambda **kwargs: kwargs)
    module = types.SimpleNamespace(SamplingParams=lambda **kwargs: kwargs, sampling_params=sampling_params)
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)


def _provider(
    text: str = '{"a": 1}', finish_reason: str | None = "stop", parser: bool = False, steps=None,
) -> VllmTextProvider:
    provider = VllmTextProvider(ModelConfig(
        id="qwen3.5-4b", display_name="Qwen", category="text", provider_class="VllmTextProvider",
        model={"hub_id": "cyankiwi/Qwen3.5-4B-AWQ-4bit"},
    ))
    provider._engine = _Engine(steps or [(text, [1, 2, 3])], finish_reason)
    provider._reasoning = _Parser() if parser else None
    provider.prompts = []

    def build_prompt(messages, thinking=True, images=None):
        provider.prompts.append(messages)
        return "prompt"

    provider._build_prompt = build_prompt
    return provider


MESSAGES = [{"role": "user", "content": "A tale as JSON."}]
SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}


async def test_json_object_holds_the_answer_to_the_json_grammar(vllm):
    provider = _provider()

    await provider.generate(MESSAGES, response_format="json_object", thinking=False)

    assert provider._engine.sampling["structured_outputs"] == {"json_object": True}


async def test_thinking_without_a_reasoning_parser_leaves_the_think_block_free_of_the_grammar(vllm):
    provider = _provider()

    await provider.generate(MESSAGES, response_format="json_object", thinking=True)

    assert provider._engine.sampling["structured_outputs"] is None


async def test_a_reasoning_parser_lets_the_grammar_hold_the_answer_after_thinking(vllm):
    provider = _provider(parser=True)

    await provider.generate(MESSAGES, response_format="json_object", thinking=True)

    assert provider._engine.sampling["structured_outputs"] == {"json_object": True}


async def test_json_schema_holds_the_answer_to_the_schema_and_shows_it_to_the_model(vllm):
    provider = _provider()

    await provider.generate(MESSAGES, response_format="json_schema", json_schema=SCHEMA, thinking=False)

    assert provider._engine.sampling["structured_outputs"] == {"json": SCHEMA}
    assert json.dumps(SCHEMA) in provider.prompts[0][0]["content"]


async def test_plain_text_has_no_grammar(vllm):
    provider = _provider()

    await provider.generate(MESSAGES, thinking=False)

    assert provider._engine.sampling["structured_outputs"] is None


async def test_reasoning_comes_apart_from_the_answer(vllm):
    result = await _provider('Plan.</think>{"a": 1}', parser=True).generate(MESSAGES, thinking=True)

    message = result["choices"][0]["message"]
    assert message["reasoning_content"] == "Plan."
    assert message["content"] == '{"a": 1}'


async def test_reasoning_cut_at_max_tokens_leaves_the_answer_empty(vllm):
    result = await _provider("Plan, then more plan", "length", parser=True).generate(MESSAGES, thinking=True)

    message = result["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "", "reasoning_content": "Plan, then more plan"}
    assert result["choices"][0]["finish_reason"] == "length"


async def test_without_thinking_the_whole_reply_is_the_answer(vllm):
    result = await _provider('{"a": 1}', parser=True).generate(MESSAGES, thinking=False)

    assert result["choices"][0]["message"] == {"role": "assistant", "content": '{"a": 1}'}


@pytest.mark.parametrize(("effort", "budget"), [("minimal", 128), ("low", 512), ("medium", 2048), ("high", None)])
async def test_reasoning_effort_sets_the_thinking_budget(vllm, effort, budget):
    provider = _provider(parser=True)

    await provider.generate(MESSAGES, thinking=True, reasoning_effort=effort)

    assert provider._engine.sampling["thinking_token_budget"] == budget


async def test_reasoning_effort_without_a_reasoning_parser_sets_no_budget(vllm):
    provider = _provider()

    await provider.generate(MESSAGES, thinking=True, reasoning_effort="low")

    assert provider._engine.sampling["thinking_token_budget"] is None


async def test_an_answer_cut_at_max_tokens_finishes_with_length(vllm):
    result = await _provider(finish_reason="length").generate(MESSAGES, thinking=False)

    assert result["choices"][0]["finish_reason"] == "length"


async def test_the_stream_ends_with_the_engine_finish_reason(vllm):
    provider = _provider(finish_reason="length")

    chunks = [c async for c in provider.generate_stream(MESSAGES, response_format="json_object", thinking=False)]

    final = json.loads(chunks[-2].removeprefix("data: "))
    assert final["choices"][0]["finish_reason"] == "length"
    assert provider._engine.sampling["structured_outputs"] == {"json_object": True}


async def test_the_stream_sends_reasoning_apart_from_the_answer(vllm):
    steps = [("Plan", [1]), ("Plan.", [1, 2]), ("Plan.</think>", [1, 2, END]), ('Plan.</think>{"a"', [1, 2, END, 3])]
    provider = _provider(parser=True, steps=steps)

    chunks = [c async for c in provider.generate_stream(MESSAGES, thinking=True)]

    deltas = [json.loads(c.removeprefix("data: "))["choices"][0]["delta"] for c in chunks[:-2]]
    assert deltas == [{"reasoning_content": "Plan"}, {"reasoning_content": "."}, {"content": '{"a"'}]
