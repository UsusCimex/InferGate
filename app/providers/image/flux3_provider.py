from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import gc
import io
import logging
from typing import Any

from app.providers.base import ImageProvider
from app.providers.image._compel import strip_weight_syntax
from app.providers.image._size import apply_size
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)

_GPU_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="flux3")
_IGNORED = ("response_format", "n")
# The step-distilled model has no classifier-free guidance and no img2img strength.
_UNSUPPORTED = ("negative_prompt", "guidance_scale", "scheduler", "image", "mask", "denoising_strength",
                "highres_fix", "refiner_switch_at", "loras", "textual_inversions")
_TRANSFORMER_VARIANTS = ("fp8r", "nvfp4")
_TEXT_ENCODER_QUANTIZATIONS = ("nf4",)
_VAE_DTYPES = ("float32", "bfloat16")


class _PinnedPager:
    """Keeps a module's tensors in pinned RAM and on the GPU only between `page_in` and `page_out`.

    `page_out` drops the GPU copies instead of copying back, so a round trip is one fast host-to-device copy.
    A bitsandbytes weight moves its quantization state in place, so that small state stays on the GPU.
    A tensor shared by two modules (tied embeddings) is copied once and stays shared.
    """

    def __init__(self, module: Any) -> None:
        import torch

        module.to("cpu")
        self._slots = []
        pinned = set()
        for sub in module.modules():
            for store in (sub._parameters, sub._buffers):
                for name, tensor in store.items():
                    if tensor is None:
                        continue
                    if id(tensor) not in pinned:
                        tensor.data = tensor.data.pin_memory()
                        pinned.add(id(tensor))
                    self._slots.append((store, name, tensor))
        torch.cuda.empty_cache()

    def page_in(self) -> None:
        import torch

        copies: dict[int, Any] = {}
        for store, name, tensor in self._slots:
            if id(tensor) not in copies:
                moved = tensor.to("cuda", non_blocking=True)
                if isinstance(tensor, torch.nn.Parameter) and not isinstance(moved, torch.nn.Parameter):
                    moved = torch.nn.Parameter(moved, requires_grad=False)
                copies[id(tensor)] = moved
            store[name] = copies[id(tensor)]
        torch.cuda.synchronize()

    def page_out(self) -> None:
        import torch

        for store, name, tensor in self._slots:
            store[name] = tensor
        torch.cuda.empty_cache()


@register_provider
class Flux3ImageProvider(ImageProvider):
    """FLUX 3 Image Klein (diffusers modular pipeline): text-to-image and reference images.

    The 9B prompt upsampler is not loaded: the prompt, without compel weights, goes to the text encoder,
    so a client sends the full scene caption the model is trained on.
    """

    def __init__(self, config):
        super().__init__(config)
        self._pipeline: Any = None
        self._text_pipeline: Any = None
        self._pager: _PinnedPager | None = None

    async def load(self, model_dir: str) -> None:
        loop = asyncio.get_running_loop()
        self._pipeline = await loop.run_in_executor(_GPU_EXECUTOR, self._load, model_dir)
        if self.config.model.get("warmup", True):
            await loop.run_in_executor(_GPU_EXECUTOR, self._warmup)
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    def _load(self, model_dir: str) -> Any:
        import torch
        from diffusers import ModularPipeline

        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()

        model = self.config.model
        hub_id = model["hub_id"]
        revision = model.get("revision") or "main"
        variant = (model.get("transformer_variant") or "").lower() or None
        if variant is not None and variant not in _TRANSFORMER_VARIANTS:
            raise ValueError(f"{self.model_id}: transformer_variant must be one of {', '.join(_TRANSFORMER_VARIANTS)}")
        encoder_quant = (model.get("text_encoder_quantization") or "").lower() or None
        if encoder_quant is not None and encoder_quant not in _TEXT_ENCODER_QUANTIZATIONS:
            raise ValueError(
                f"{self.model_id}: text_encoder_quantization must be one of {', '.join(_TEXT_ENCODER_QUANTIZATIONS)}"
            )
        vae_dtype = str(model.get("vae_dtype") or "float32").lower()
        if vae_dtype not in _VAE_DTYPES:
            raise ValueError(f"{self.model_id}: vae_dtype must be one of {', '.join(_VAE_DTYPES)}")
        page_text_encoder = bool(model.get("page_text_encoders", False))

        logger.info(
            "Loading %s from %s (transformer %s, text encoder %s%s, VAE %s)",
            self.model_id, hub_id, variant or "bf16", encoder_quant or "bf16",
            " paged to RAM" if page_text_encoder else "", vae_dtype,
        )
        blocks = ModularPipeline.from_pretrained(hub_id, cache_dir=model_dir, revision=revision).blocks
        blocks.sub_blocks.pop("prompt_upsampler", None)
        # A pipeline of its own for the prompt, so the text encoder can wait in RAM while the transformer works.
        text_pipe = blocks.sub_blocks.pop("text_encoder").init_pipeline(hub_id)
        pipe = blocks.init_pipeline(hub_id)

        text_pipe.load_components(
            names=["tokenizer"] if encoder_quant is not None else None,
            dtype=torch.bfloat16, cache_dir=model_dir, revision=revision,
        )
        if encoder_quant is not None:
            text_pipe.update_components(text_encoder=self._nf4_text_encoder(hub_id, revision, model_dir))
        if page_text_encoder:
            self._pager = _PinnedPager(text_pipe.text_encoder)
        else:
            text_pipe.text_encoder.to("cuda")

        pipe.load_components(
            names=["vae"] if variant is not None else None,
            dtype={"vae": getattr(torch, vae_dtype), "default": torch.bfloat16}, cache_dir=model_dir, revision=revision,
        )
        if variant is not None:
            pipe.update_components(transformer=self._quantized_transformer(hub_id, revision, variant, model_dir))
        pipe.to("cuda")
        if torch.cuda.is_available():
            _flush_cache_before_decode(pipe.vae, torch)
        self._text_pipeline = text_pipe
        return pipe

    @staticmethod
    def _quantized_transformer(hub_id: str, revision: str, variant: str, model_dir: str) -> Any:
        import torch
        from diffusers import BFLQuantizationConfig, Flux3Transformer2DModel

        return Flux3Transformer2DModel.from_single_file(
            f"https://huggingface.co/{hub_id}/blob/{revision}/variants/{variant}/model.safetensors",
            config=hub_id,
            subfolder="transformer",
            quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
            dtype=torch.bfloat16,
            cache_dir=model_dir,
        )

    @staticmethod
    def _nf4_text_encoder(hub_id: str, revision: str, model_dir: str) -> Any:
        import torch
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        return Qwen3VLForConditionalGeneration.from_pretrained(
            hub_id, subfolder="text_encoder", revision=revision, quantization_config=config,
            dtype=torch.bfloat16, device_map={"": 0}, cache_dir=model_dir,
        )

    def _warmup(self) -> None:
        try:
            self._run({"prompt": "warmup", "width": 256, "height": 256, "num_inference_steps": 1, "output": "images"})
        except Exception as e:
            logger.warning("Warmup failed for %s: %s", self.model_id, e)

    async def unload(self) -> None:
        import torch

        self._pipeline = None
        self._text_pipeline = None
        self._pager = None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, gc.collect)
        if torch.cuda.is_available():
            def _cuda_cleanup() -> None:
                for fn in (torch.cuda.synchronize, torch.cuda.empty_cache, torch.cuda.ipc_collect):
                    with contextlib.suppress(Exception):
                        fn()
            await loop.run_in_executor(None, _cuda_cleanup)
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def generate(self, prompt: str, **params: Any) -> bytes:
        call = self._call_kwargs(prompt, params)
        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(_GPU_EXECUTOR, self._run, call)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def _run(self, call: dict[str, Any]) -> Any:
        import torch

        call = dict(call)
        call["prompt_embeds"] = self._encode(call.pop("prompt"))
        try:
            return self._pipeline(**call)[0]
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _encode(self, prompt: str) -> Any:
        if self._pager is None:
            return self._text_pipeline(prompt=prompt, output="prompt_embeds")
        self._pager.page_in()
        try:
            return self._text_pipeline(prompt=prompt, output="prompt_embeds")
        finally:
            self._pager.page_out()

    def _call_kwargs(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        """Pipeline arguments; ValueError for what the distilled pipeline cannot do."""
        # An empty scheduler or list from a client that serves several models means the model's default.
        unsupported = sorted(key for key in _UNSUPPORTED if params.get(key) not in (None, "", [], {}))
        if unsupported:
            raise ValueError(f"{self.model_id} does not take {', '.join(unsupported)}")
        call = dict(self.config.model.get("default_params", {}))
        call.update(params)
        apply_size(call)
        for key in (*_IGNORED, *_UNSUPPORTED):
            call.pop(key, None)
        references = call.pop("reference_images", None) or []
        if references:
            # The pipeline keeps each reference at its own size, and a 1024 px portrait doubles the tokens of every step.
            max_side = self.config.model.get("reference_max_side")
            call["image"] = [_decode_image(r, max_side) for r in references]
        seed = call.pop("seed", None)
        if seed is not None:
            import torch
            # The reference implementation draws its noise on the CPU, so a seed gives the same picture on any GPU.
            call["generator"] = torch.Generator().manual_seed(int(seed))
        call["prompt"] = strip_weight_syntax(prompt)
        call["output"] = "images"
        return call


def _flush_cache_before_decode(vae: Any, torch: Any) -> None:
    """The denoising steps leave ~2 GB of freed activations in the allocator; without them the decode peak fits 12 GB."""
    decode = vae.decode

    def flushed(*args: Any, **kwargs: Any) -> Any:
        torch.cuda.empty_cache()
        return decode(*args, **kwargs)

    vae.decode = flushed


def _decode_image(b64_str: str, max_side: int | None = None) -> Any:
    from PIL import Image, UnidentifiedImageError

    payload = b64_str.split(",", 1)[1] if b64_str.startswith("data:") else b64_str
    try:
        image = Image.open(io.BytesIO(base64.b64decode(payload, validate=False)))
        image.load()
    except (ValueError, UnidentifiedImageError) as e:
        raise ValueError("reference_images must be base64 PNG or JPEG") from e
    image = image.convert("RGB")
    if max_side:
        image.thumbnail((int(max_side), int(max_side)), Image.LANCZOS)
    return image
