from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.providers.base import TextProvider
from app.providers.registry import register_provider
from app.providers.text._chat_images import (
    decode_data_url,
    image_urls,
    text_of,
    with_system_instruction,
)

logger = logging.getLogger(__name__)


_JSON_FORMATS = ("json_object", "json_schema")
# Thinking tokens per OpenAI reasoning_effort; high and xhigh leave the length to the model.
_EFFORT_BUDGETS = {"minimal": 128, "low": 512, "medium": 2048}


@register_provider
class VllmTextProvider(TextProvider):
    """vLLM-backed LLM provider with streaming and optional thinking mode."""

    def __init__(self, config):
        super().__init__(config)
        self._engine = None
        self._tokenizer = None
        self._reasoning = None
        self._sleeping = False

    @property
    def sleep_level(self) -> int:
        """vLLM sleep mode on eviction: 1 keeps the weights in RAM, 2 reads them again; 0 shuts the engine down."""
        return int(self.config.model.get("sleep_level") or 0)

    async def load(self, model_dir: str) -> None:
        if self._sleeping:
            try:
                await self._wake_up()
                return
            except Exception as e:
                logger.warning("Waking %s failed, starting the engine again: %s", self.model_id, e)
                await self.release()

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
        reasoning_parser = self.config.model.get("reasoning_parser")
        reasoning: dict[str, Any] = {}
        if reasoning_parser:
            from vllm.config import ReasoningConfig

            reasoning = {"reasoning_parser": reasoning_parser, "reasoning_config": ReasoningConfig()}
        multimodal = {
            key: self.config.model[key]
            for key in ("limit_mm_per_prompt", "mm_processor_kwargs")
            if key in self.config.model
        }

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
            enable_sleep_mode=self.sleep_level > 0,
            **multimodal,
            **reasoning,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        try:
            self._tokenizer = self._engine.get_tokenizer()
        except Exception:
            self._tokenizer = None
            logger.warning("Could not get tokenizer for %s, falling back to ChatML", self.model_id)
        if reasoning_parser and self._tokenizer is not None:
            from vllm.reasoning import ReasoningParserManager

            self._reasoning = ReasoningParserManager.get_reasoning_parser(reasoning_parser)(self._tokenizer)

        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def _wake_up(self) -> None:
        started = time.monotonic()
        if self.sleep_level == 2:
            await self._engine.wake_up(tags=["weights"])
            await self._engine.collective_rpc("reload_weights")
            await self._engine.wake_up(tags=["kv_cache"])
        else:
            await self._engine.wake_up()
        self._sleeping = False
        self._loaded = True
        logger.info("Woke %s in %.1fs", self.model_id, time.monotonic() - started)

    async def unload(self) -> None:
        if self._sleeping:
            return
        if self._engine is not None and self.sleep_level:
            started = time.monotonic()
            await self._engine.sleep(self.sleep_level)
            self._sleeping = True
            self._loaded = False
            logger.info("%s sleeps at level %d (%.1fs)", self.model_id, self.sleep_level,
                        time.monotonic() - started)
            return
        await self.release()

    async def release(self) -> None:
        import gc

        import torch

        self._sleeping = False
        if self._engine:
            if hasattr(self._engine, "shutdown"):
                self._engine.shutdown()
            del self._engine
            self._engine = None
        self._tokenizer = None
        self._reasoning = None

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, gc.collect)
        if torch.cuda.is_available():
            def _cuda_cleanup() -> None:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            await loop.run_in_executor(None, _cuda_cleanup)
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def _prepare(self, messages: list[dict], params: dict[str, Any]):
        from vllm import SamplingParams

        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        max_tokens = defaults.pop("max_tokens", 4096)
        response_format = defaults.pop("response_format", None)
        schema = defaults.pop("json_schema", None)
        thinking = defaults.pop("thinking", True)
        effort = defaults.pop("reasoning_effort", None)
        budget = _EFFORT_BUDGETS.get(effort) if thinking and self._reasoning is not None else None

        # Thinking mode needs headroom for the <think> block.
        if thinking and max_tokens < 4096:
            max_tokens = 4096

        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=defaults.pop("temperature", 0.7),
            top_p=defaults.pop("top_p", 0.9),
            structured_outputs=self._grammar(response_format, schema, thinking),
            thinking_token_budget=budget,
        )

        if response_format == "json_schema" and schema is not None:
            messages = with_system_instruction(
                messages, f"Respond with JSON that matches this JSON schema: {json.dumps(schema)}"
            )
        elif response_format in _JSON_FORMATS:
            messages = with_system_instruction(messages, "Respond with valid JSON only.")

        images = await asyncio.to_thread(self._decode_images, messages)
        prompt = self._build_prompt(messages, thinking=thinking, images=images)
        return prompt, sampling, thinking and self._reasoning is not None

    def _grammar(self, response_format: str | None, schema: dict | None, thinking: bool):
        """The JSON grammar holds the answer; without a reasoning parser it would hold the <think> block too, so
        thinking then leaves only the instruction."""
        if response_format not in _JSON_FORMATS or (thinking and self._reasoning is None):
            return None
        from vllm.sampling_params import StructuredOutputsParams

        if response_format == "json_schema" and schema is not None:
            return StructuredOutputsParams(json=schema)
        return StructuredOutputsParams(json_object=True)

    async def generate(self, messages: list[dict], **params: Any) -> dict:
        prompt, sampling, split = await self._prepare(messages, params)
        request_id = str(uuid.uuid4())

        async for output in self._engine.generate(prompt, sampling, request_id):
            final = output
        text = ""
        completion_tokens = 0
        finish_reason = "stop"
        if final.outputs:
            text = final.outputs[0].text
            completion_tokens = len(final.outputs[0].token_ids)
            finish_reason = final.outputs[0].finish_reason or "stop"
        prompt_tokens = len(final.prompt_token_ids) if final.prompt_token_ids else 0

        message: dict[str, Any] = {"role": "assistant", "content": text}
        if split:
            reasoning, content = self._reasoning.extract_reasoning(text, None)
            message.update(content=content or "", reasoning_content=reasoning)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def generate_stream(self, messages: list[dict], **params: Any) -> AsyncIterator[str]:
        """Stream chat completion as SSE chunks."""
        prompt, sampling, split = await self._prepare(messages, params)
        request_id = str(uuid.uuid4())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def chunk(delta: dict[str, Any], finish_reason: str | None) -> str:
            body = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(body)}\n\n"

        text = ""
        token_ids: list[int] = []
        finish_reason = "stop"
        try:
            async for output in self._engine.generate(prompt, sampling, request_id):
                if not output.outputs:
                    continue
                out = output.outputs[0]
                finish_reason = out.finish_reason or finish_reason
                current_ids = list(out.token_ids)
                delta_text = out.text[len(text):]
                if split:
                    parsed = self._reasoning.extract_reasoning_streaming(
                        text, out.text, delta_text, token_ids, current_ids, current_ids[len(token_ids):]
                    )
                    delta = {} if parsed is None else {
                        key: value
                        for key, value in (("reasoning_content", parsed.reasoning), ("content", parsed.content))
                        if value
                    }
                else:
                    delta = {"content": delta_text} if delta_text else {}
                text, token_ids = out.text, current_ids
                if delta:
                    yield chunk(delta, None)
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected: abort so the GPU frees immediately.
            await self._abort_request(request_id)
            raise

        yield chunk({}, finish_reason)
        yield "data: [DONE]\n\n"

    async def _abort_request(self, request_id: str) -> None:
        """Best-effort abort of an in-flight vLLM request."""
        if self._engine is None:
            return
        abort = getattr(self._engine, "abort", None) or getattr(self._engine, "abort_request", None)
        if abort is None:
            return
        try:
            result = abort(request_id)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.debug("Failed to abort vLLM request %s: %s", request_id, e)

    @staticmethod
    def _decode_images(messages: list[dict]) -> list:
        return [decode_data_url(url) for url in image_urls(messages)]

    def _build_prompt(self, messages: list[dict], thinking: bool = True, images: list | None = None):
        """Build a vLLM prompt from chat messages (native template or ChatML fallback)."""
        from vllm import TextPrompt, TokensPrompt

        if images:
            text = None
            if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
                text = self._apply_template(messages, thinking, tokenize=False)
            if text is None:
                raise ValueError(f"model '{self.model_id}' could not render images in its chat template")
            return TextPrompt(prompt=text, multi_modal_data={"image": images})

        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            token_ids = self._apply_template(messages, thinking)
            if token_ids is not None:
                return TokensPrompt(prompt_token_ids=token_ids)

        text = self._format_chatml(messages, thinking)
        if self._tokenizer:
            return TokensPrompt(prompt_token_ids=self._tokenizer.encode(text))
        return text

    def _apply_template(self, messages: list[dict], thinking: bool, tokenize: bool = True):
        """Apply the tokenizer's chat template; returns None if it can't be applied."""
        kwargs: dict[str, Any] = {"tokenize": tokenize, "add_generation_prompt": True}
        if not thinking:
            kwargs["enable_thinking"] = False
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Retry without enable_thinking for templates that don't accept it.
            kwargs.pop("enable_thinking", None)
            try:
                return self._tokenizer.apply_chat_template(messages, **kwargs)
            except Exception:
                return None
        except Exception:
            return None

    @staticmethod
    def _format_chatml(messages: list[dict], thinking: bool) -> str:
        """Format messages as raw ChatML for models without a chat template."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = text_of(msg.get("content", ""))
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if thinking:
            parts.append("<|im_start|>assistant\n")
        else:
            parts.append("<|im_start|>assistant\n/no_think\n")
        return "\n".join(parts)


