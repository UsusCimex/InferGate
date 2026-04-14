from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

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

        # Get tokenizer for proper chat template formatting
        try:
            self._tokenizer = await self._engine.get_tokenizer()
        except Exception:
            self._tokenizer = None
            logger.warning("Could not get tokenizer for %s, falling back to ChatML", self.model_id)

        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def unload(self) -> None:
        import gc

        import torch

        if self._engine:
            if hasattr(self._engine, "shutdown"):
                self._engine.shutdown()
            del self._engine
            self._engine = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
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

    async def generate_stream(self, messages: list[dict], **params: Any) -> AsyncIterator[str]:
        """Stream chat completion as SSE chunks."""
        from vllm import SamplingParams

        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        max_tokens = defaults.pop("max_tokens", 4096)
        temperature = defaults.pop("temperature", 0.7)
        top_p = defaults.pop("top_p", 0.9)
        response_format = defaults.pop("response_format", None)
        thinking = defaults.pop("thinking", True)

        if thinking and max_tokens < 4096:
            max_tokens = 4096

        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

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
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        prev_text_len = 0
        async for output in self._engine.generate(prompt, sampling, request_id):
            if output.outputs:
                new_text = output.outputs[0].text[prev_text_len:]
                prev_text_len = len(output.outputs[0].text)
                if new_text:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": new_text},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk with finish_reason
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    def _format_messages(self, messages: list[dict], thinking: bool = True) -> str:
        """Format chat messages using tokenizer's chat template when available.
        Falls back to ChatML format for models without a template.
        """
        # Use tokenizer's built-in chat template (works for Llama, Mistral, Qwen, etc.)
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass  # Fall through to manual ChatML

        # Fallback: manual ChatML format
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


