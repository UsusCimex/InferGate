"""vLLM sleep mode: eviction puts the engine to sleep, the next load wakes it, a reload shuts it down."""
from __future__ import annotations

import sys
import types

import pytest

from app.config import ModelConfig
from app.providers.text.vllm_provider import VllmTextProvider


class _FakeEngine:
    def __init__(self, args):
        self.args = args
        self.calls = []

    def get_tokenizer(self):
        return None

    async def sleep(self, level):
        self.calls.append(("sleep", level))

    async def wake_up(self, tags=None):
        self.calls.append(("wake_up", tags))

    async def collective_rpc(self, method):
        self.calls.append(("rpc", method))

    def shutdown(self):
        self.calls.append(("shutdown",))


@pytest.fixture
def engines(monkeypatch):
    created = []

    def from_engine_args(args):
        created.append(_FakeEngine(args))
        return created[-1]

    vllm = types.SimpleNamespace(
        AsyncEngineArgs=lambda **kwargs: kwargs,
        AsyncLLMEngine=types.SimpleNamespace(from_engine_args=from_engine_args),
    )
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return created


def _provider(sleep_level: int) -> VllmTextProvider:
    return VllmTextProvider(ModelConfig(
        id="qwen3.5-4b", display_name="Qwen", category="text", provider_class="VllmTextProvider",
        model={"hub_id": "cyankiwi/Qwen3.5-4B-AWQ-4bit", "sleep_level": sleep_level},
    ))


@pytest.mark.asyncio
async def test_eviction_puts_the_engine_to_sleep_and_the_next_load_wakes_it(engines):
    provider = _provider(1)
    await provider.load("/models")
    await provider.unload()

    assert not provider.is_loaded()
    await provider.load("/models")

    assert provider.is_loaded()
    assert len(engines) == 1
    assert engines[0].args["enable_sleep_mode"] is True
    assert engines[0].calls == [("sleep", 1), ("wake_up", None)]


@pytest.mark.asyncio
async def test_level_two_reads_the_weights_again_on_waking(engines):
    provider = _provider(2)
    await provider.load("/models")
    await provider.unload()
    await provider.load("/models")

    assert engines[0].calls == [("sleep", 2), ("wake_up", ["weights"]), ("rpc", "reload_weights"),
                                ("wake_up", ["kv_cache"])]


@pytest.mark.asyncio
async def test_a_sleeping_engine_is_shut_down_on_release(engines):
    provider = _provider(1)
    await provider.load("/models")
    await provider.unload()
    await provider.unload()
    await provider.release()

    assert engines[0].calls == [("sleep", 1), ("shutdown",)]
    await provider.load("/models")
    assert len(engines) == 2


@pytest.mark.asyncio
async def test_without_sleep_mode_eviction_shuts_the_engine_down(engines):
    provider = _provider(0)
    await provider.load("/models")
    await provider.unload()

    assert engines[0].args["enable_sleep_mode"] is False
    assert engines[0].calls == [("shutdown",)]


@pytest.mark.asyncio
async def test_a_failed_wake_up_starts_a_new_engine(engines):
    provider = _provider(1)
    await provider.load("/models")
    await provider.unload()

    async def broken(tags=None):
        raise RuntimeError("CUDA out of memory")

    engines[0].wake_up = broken
    await provider.load("/models")

    assert provider.is_loaded()
    assert len(engines) == 2
    assert ("shutdown",) in engines[0].calls
