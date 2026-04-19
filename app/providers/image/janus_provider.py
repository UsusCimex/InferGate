"""Autoregressive text-to-image provider for DeepSeek's Janus-Pro series.

Janus generates a 384×384 image by sampling 576 discrete image tokens (24×24
grid) sequentially from a causal LLM, then decoding them through a built-in
LlamaGen VQ decoder. No diffusion, no offload, no scheduler — just AR sampling
+ classifier-free guidance.

Install path for the worker:
    pip install git+https://github.com/deepseek-ai/Janus.git
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import io
import logging
from typing import Any

from app.providers.base import ImageProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)

# Serialise generation: AR decoding keeps past_key_values state across steps
# that must not interleave between concurrent requests.
_GPU_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="janus-gpu"
)

# Janus-Pro fixed generation geometry (img_size / patch_size = 384 / 16 = 24).
_IMAGE_TOKEN_NUM = 576  # 24 × 24
_VQ_SHAPE = [1, 8, 24, 24]  # single image, 8 VQ channels, 24×24 tokens


@register_provider
class JanusImageProvider(ImageProvider):
    """Autoregressive image generator (deepseek-ai/Janus-Pro-1B / 7B).

    Native output is 384×384 — the `size` / `width` / `height` params from
    the OpenAI-compatible schema are ignored (no built-in upscaler).
    Accepts `cfg_weight` (default 5.0), `temperature` (default 1.0), `seed`.
    """

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._tokenizer = None

    async def load(self, model_dir: str) -> None:
        import torch
        from janus.models import VLChatProcessor  # noqa: F401 — validates install
        from transformers import AutoModelForCausalLM

        hub_id = self.config.model["hub_id"]
        dtype_name = self.config.model.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_name)

        # Blackwell race guard: force CUDA context init before first alloc.
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()

        logger.info("Loading %s from %s", self.model_id, hub_id)
        loop = asyncio.get_running_loop()

        def _load():
            processor = VLChatProcessor.from_pretrained(hub_id, cache_dir=model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                hub_id,
                cache_dir=model_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            return processor, model.to(dtype).cuda().eval()

        self._processor, self._model = await loop.run_in_executor(_GPU_EXECUTOR, _load)
        self._tokenizer = self._processor.tokenizer
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def unload(self) -> None:
        import gc

        import torch

        self._model = None
        self._processor = None
        self._tokenizer = None

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, gc.collect)
        if torch.cuda.is_available():
            def _cleanup() -> None:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            await loop.run_in_executor(None, _cleanup)
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def generate(self, prompt: str, **params: Any) -> bytes:
        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        # Drop OpenAI-schema params we don't honour (native 384×384 only).
        for k in ("size", "width", "height", "response_format", "n"):
            defaults.pop(k, None)

        cfg_weight = float(defaults.get("cfg_weight", 5.0))
        temperature = float(defaults.get("temperature", 1.0))
        seed = defaults.get("seed")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _GPU_EXECUTOR, self._sample, prompt, cfg_weight, temperature, seed
        )

    def _sample(
        self,
        prompt: str,
        cfg_weight: float,
        temperature: float,
        seed: int | None,
    ) -> bytes:
        import numpy as np
        import torch
        from PIL import Image

        if seed is not None:
            torch.manual_seed(int(seed))

        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self._processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self._processor.sft_format,
            system_prompt="",
        )
        full_prompt = sft_format + self._processor.image_start_tag

        input_ids = torch.LongTensor(self._tokenizer.encode(full_prompt)).cuda()

        # CFG batch: row 0 = conditional (real prompt), row 1 = unconditional
        # (prompt tokens replaced by pad_id between BOS/EOS).
        tokens = torch.zeros((2, len(input_ids)), dtype=torch.int).cuda()
        tokens[0] = input_ids
        tokens[1] = input_ids
        tokens[1, 1:-1] = self._processor.pad_id

        inputs_embeds = self._model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((1, _IMAGE_TOKEN_NUM), dtype=torch.int).cuda()
        past_key_values = None

        with torch.no_grad():
            for i in range(_IMAGE_TOKEN_NUM):
                out = self._model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values

                logits = self._model.gen_head(out.last_hidden_state[:, -1, :])
                cond, uncond = logits[0::2, :], logits[1::2, :]
                logits = uncond + cfg_weight * (cond - uncond)
                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                # Duplicate next token for both CFG rows on the next step.
                paired = torch.cat(
                    [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
                ).view(-1)
                inputs_embeds = self._model.prepare_gen_img_embeds(paired).unsqueeze(dim=1)

            dec = self._model.gen_vision_model.decode_code(
                generated_tokens.to(torch.int), shape=_VQ_SHAPE
            )

        # VQ decoder yields (B, 3, 384, 384) in [-1, 1]. Convert to uint8 RGB.
        arr = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        arr = np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr[0])
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
