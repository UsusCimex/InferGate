from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from app.providers.base import TextProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class VllmTextProvider(TextProvider):
    """Universal provider for LLMs via vLLM engine."""

    def __init__(self, config):
        super().__init__(config)
        self._engine = None
        self._tokenizer = None

    async def load(self, model_dir: str) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        hub_id = self.config.model["hub_id"]
        dtype_name = self.config.model.get("torch_dtype", "float16")
        context_length = self.config.model.get("context_length", 32768)

        logger.info("Loading %s from %s", self.model_id, hub_id)
        gpu_memory = self.config.model.get("gpu_memory_utilization", 0.9)
        enforce_eager = self.config.model.get("enforce_eager", False)
        quantization = self.config.model.get("quantization", None)
        kv_cache_dtype = self.config.model.get("kv_cache_dtype", "auto")
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        engine_args = AsyncEngineArgs(
            model=hub_id,
            download_dir=model_dir,
            dtype=dtype_name,
            max_model_len=context_length,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory,
            enforce_eager=enforce_eager,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def unload(self) -> None:
        import gc

        import torch

        if self._engine:
            del self._engine
            self._engine = None
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def generate(self, messages: list[dict], **params: Any) -> dict:
        from vllm import SamplingParams

        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        max_tokens = defaults.pop("max_tokens", 4096)
        temperature = defaults.pop("temperature", 0.7)
        top_p = defaults.pop("top_p", 0.9)
        response_format = defaults.pop("response_format", None)
        thinking = defaults.pop("thinking", True)

        # With thinking enabled, model needs more tokens for <think> block
        if thinking and max_tokens < 4096:
            max_tokens = 4096

        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # JSON mode: add instruction to system prompt
        messages = list(messages)
        if response_format == "json_object":
            if messages and messages[0].get("role") == "system":
                messages[0] = {
                    **messages[0],
                    "content": messages[0]["content"] + "\nRespond with valid JSON only.",
                }
            else:
                messages.insert(0, {"role": "system", "content": "Respond with valid JSON only."})

        prompt = self._format_messages(messages, thinking=thinking)
        request_id = str(uuid.uuid4())

        output_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        async for output in self._engine.generate(prompt, sampling, request_id):
            final = output
        if final.outputs:
            output_text = final.outputs[0].text
            completion_tokens = len(final.outputs[0].token_ids)
        prompt_tokens = len(final.prompt_token_ids) if final.prompt_token_ids else 0

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _format_messages(self, messages: list[dict], thinking: bool = True) -> str:
        """Format chat messages into ChatML prompt.
        Adds /no_think tag when thinking is disabled.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if thinking:
            parts.append("<|im_start|>assistant\n")
        else:
            parts.append("<|im_start|>assistant\n/no_think\n")
        return "\n".join(parts)


