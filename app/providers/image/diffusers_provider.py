from __future__ import annotations

import asyncio
import concurrent.futures
import io
import logging
from typing import Any

from app.providers.base import ImageProvider
from app.providers.image._compel import CompelAdapter, has_weight_syntax
from app.providers.image._highres_fix import apply_highres_fix
from app.providers.image._lora import LoraCache
from app.providers.image._schedulers import maybe_swap_scheduler
from app.providers.image._textual_inversion import TextualInversionRegistry
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)

_GPU_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="gpu-inference"
)


def _disable_caching_allocator_warmup() -> None:
    """Neutralise diffusers/transformers caching-allocator warmup.

    Both libraries pre-allocate a giant CUDA tensor inside `from_pretrained`
    to speed up subsequent weight loads. On Blackwell (sm_120) this races
    with lazy CUDA context init and raises `CUDA driver error: device not
    ready` (cudaErrorNotReady, 600). The warmup is a perf hint, not a
    correctness primitive — no-opping it makes the first load a few seconds
    slower but removes the race entirely.

    `from X import Y` creates a local binding, so patching X.Y alone is
    not enough; we patch every module that re-imported the symbol.
    """
    noop = lambda *_a, **_kw: None  # noqa: E731
    import importlib

    targets = [
        ("diffusers.models.model_loading_utils", "_caching_allocator_warmup"),
        ("diffusers.models.modeling_utils", "_caching_allocator_warmup"),
        ("transformers.modeling_utils", "caching_allocator_warmup"),
    ]
    for module_path, attr in targets:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        if hasattr(module, attr):
            setattr(module, attr, noop)


def _nf4_kwargs(dtype: Any) -> tuple[str, dict[str, Any]]:
    return "bitsandbytes_4bit", {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": dtype,
    }


def _fp8_kwargs(_dtype: Any) -> tuple[str, dict[str, Any]]:
    return "quanto", {"weights_dtype": "float8"}


_QUANT_BACKENDS = {
    "nf4": _nf4_kwargs,
    "int4": _nf4_kwargs,
    "fp8": _fp8_kwargs,
    "float8": _fp8_kwargs,
}


@register_provider
class DiffusersImageProvider(ImageProvider):
    """Universal provider for any diffusers-compatible model.
    Supports FLUX, Stable Diffusion, PixArt, etc.
    The specific model is determined by the YAML config (hub_id).
    """

    def __init__(self, config):
        super().__init__(config)
        self._pipeline = None
        self._compel = CompelAdapter(self.model_id)
        self._lora = LoraCache(self.model_id)
        self._ti = TextualInversionRegistry(self.model_id)
        self._model_dir: str | None = None

        # Img2img / inpaint pipelines, built lazily via `from_pipe` so they
        # share UNet / VAE / text encoders with the base pipeline (zero VRAM
        # overhead). None until the first img2img / inpaint / highres_fix
        # request touches the respective slot.
        self._img2img_pipeline = None
        self._inpaint_pipeline = None
        # Optional SDXL refiner — loaded eagerly at load() time when
        # `model.refiner_hub_id` is set in YAML. Shares text_encoder_2
        # and VAE with the base pipeline (~2-3 GB VRAM saving).
        self._refiner = None

    async def load(self, model_dir: str) -> None:
        import torch
        from diffusers import DiffusionPipeline

        self._model_dir = model_dir

        _disable_caching_allocator_warmup()

        # Force CUDA context init before any weight loading / quantization.
        # On Blackwell (sm_120), lazy context init races with the first alloc
        # inside bitsandbytes.quantize_4bit / diffusers warmup → cudaErrorNotReady.
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()

        hub_id = self.config.model["hub_id"]
        dtype_name = self.config.model.get("torch_dtype", "float16")
        dtype = getattr(torch, dtype_name)

        kwargs: dict[str, Any] = {
            "cache_dir": model_dir,
            "torch_dtype": dtype,
        }
        if variant := self.config.model.get("variant"):
            kwargs["variant"] = variant
        if revision := self.config.model.get("revision"):
            kwargs["revision"] = revision

        # Drop T5 text encoder for SD 3.5 to save ~9.5 GB VRAM
        if self.config.model.get("drop_t5", False):
            kwargs["text_encoder_3"] = None
            kwargs["tokenizer_3"] = None

        # Optional weight-only quantization (nf4 via bitsandbytes, fp8 via quanto).
        # Quantises just the large components (transformer, T5) so the pipeline
        # fits on consumer GPUs without meaningful quality loss.
        quantization = self.config.model.get("quantization")
        if quantization:
            kwargs["quantization_config"] = self._build_quantization_config(
                quantization, dtype
            )

        cpu_offload = self.config.model.get("cpu_offload", False)
        sequential_offload = self.config.model.get("sequential_cpu_offload", False)

        logger.info("Loading %s from %s", self.model_id, hub_id)
        loop = asyncio.get_running_loop()

        def _load():
            pipe = DiffusionPipeline.from_pretrained(hub_id, **kwargs)
            if sequential_offload:
                pipe.enable_sequential_cpu_offload()
            elif cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            return pipe

        self._pipeline = await loop.run_in_executor(_GPU_EXECUTOR, _load)
        if self.config.model.get("compel", True):
            self._compel.init(self._pipeline)

        # Optional SDXL Refiner. Shares text_encoder_2 + VAE with the base
        # to avoid duplicating ~2-3 GB of VRAM. Only triggered on requests
        # that pass `refiner_switch_at`; if YAML doesn't set refiner_hub_id
        # the ensemble mode is unreachable.
        refiner_hub_id = self.config.model.get("refiner_hub_id")
        if refiner_hub_id:
            logger.info("Loading refiner %s for %s", refiner_hub_id, self.model_id)

            def _load_refiner():
                from diffusers import StableDiffusionXLImg2ImgPipeline

                ref_kwargs: dict[str, Any] = {
                    "text_encoder_2": self._pipeline.text_encoder_2,
                    "vae": self._pipeline.vae,
                    "torch_dtype": dtype,
                    "cache_dir": model_dir,
                    "use_safetensors": True,
                }
                if variant := self.config.model.get("refiner_variant", self.config.model.get("variant")):
                    ref_kwargs["variant"] = variant
                ref = StableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_hub_id, **ref_kwargs)
                if sequential_offload:
                    ref.enable_sequential_cpu_offload()
                elif cpu_offload:
                    ref.enable_model_cpu_offload()
                else:
                    ref.to("cuda")
                return ref

            self._refiner = await loop.run_in_executor(_GPU_EXECUTOR, _load_refiner)
            logger.info("Loaded refiner for %s", self.model_id)

        # Warmup: run a minimal dummy generation so cuDNN kernel tuning,
        # Triton compilation, and offload-swap patterns happen *here* instead
        # of punishing the first real client request. Controlled by YAML
        # `warmup: true|false` (default true).
        if self.config.model.get("warmup", True):
            logger.info("Warming up %s …", self.model_id)
            await loop.run_in_executor(_GPU_EXECUTOR, self._warmup)
            logger.info("Warmup complete for %s", self.model_id)

        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    def _ensure_img2img_pipe(self):
        """Lazy-build an img2img pipeline that shares weights with the base.

        Uses AutoPipelineForImage2Image.from_pipe so UNet / VAE / text
        encoders are literally the same tensors (no duplicate VRAM). The
        scheduler is its own instance per-pipe so we re-sync it on every
        request — per-request scheduler swap on the base must apply to the
        img2img pass too for consistent trajectory.
        """
        from diffusers import AutoPipelineForImage2Image

        if self._img2img_pipeline is None:
            self._img2img_pipeline = AutoPipelineForImage2Image.from_pipe(self._pipeline)
        self._img2img_pipeline.scheduler = self._pipeline.scheduler
        return self._img2img_pipeline

    def _ensure_inpaint_pipe(self):
        """Lazy inpaint pipeline via from_pipe — shares UNet/VAE/text encoders."""
        from diffusers import AutoPipelineForInpainting

        if self._inpaint_pipeline is None:
            self._inpaint_pipeline = AutoPipelineForInpainting.from_pipe(self._pipeline)
        self._inpaint_pipeline.scheduler = self._pipeline.scheduler
        return self._inpaint_pipeline

    @staticmethod
    def _decode_image(b64_str: str, mode: str | None = None):
        """Decode base64 (with/without data: prefix) to PIL.Image, raises ValueError on bad input."""
        import base64 as _b64
        import io as _io

        from PIL import Image, UnidentifiedImageError

        payload = b64_str.split(",", 1)[1] if b64_str.startswith("data:") else b64_str
        try:
            raw = _b64.b64decode(payload, validate=False)
        except Exception as e:
            raise ValueError(f"image/mask is not valid base64: {e}") from e
        try:
            img = Image.open(_io.BytesIO(raw))
            img.load()
        except UnidentifiedImageError as e:
            raise ValueError("image/mask is not a recognised PNG/JPEG") from e
        if mode is not None and img.mode != mode:
            img = img.convert(mode)
        return img

    def _warmup(self) -> None:
        """Minimal dummy inference to trigger kernel autotuning + offload hooks."""
        defaults = dict(self.config.model.get("default_params", {}))
        defaults.pop("response_format", None)
        defaults.pop("n", None)
        defaults.pop("size", None)
        defaults["num_inference_steps"] = 1
        defaults["width"] = 256
        defaults["height"] = 256
        try:
            self._pipeline(prompt="warmup", **defaults)
        except Exception as e:
            logger.warning("Warmup failed for %s: %s", self.model_id, e)

    def _build_quantization_config(self, quantization: str, dtype: Any) -> Any:
        """Translate YAML `quantization: nf4|fp8` into a diffusers pipeline config."""
        from diffusers import PipelineQuantizationConfig

        try:
            backend_builder = _QUANT_BACKENDS[quantization.lower()]
        except KeyError as e:
            raise ValueError(
                f"Unsupported quantization '{quantization}' for {self.model_id}. "
                f"Supported: {sorted(_QUANT_BACKENDS)}"
            ) from e
        components = self.config.model.get(
            "quantize_components", ["transformer", "text_encoder_2"]
        )
        backend, quant_kwargs = backend_builder(dtype)
        return PipelineQuantizationConfig(
            quant_backend=backend,
            quant_kwargs=quant_kwargs,
            components_to_quantize=components,
        )

    async def unload(self) -> None:
        import gc

        import torch

        if self._refiner is not None:
            if hasattr(self._refiner, "maybe_free_model_hooks"):
                self._refiner.maybe_free_model_hooks()
            del self._refiner
            self._refiner = None
        if self._pipeline is not None:
            if hasattr(self._pipeline, "maybe_free_model_hooks"):
                self._pipeline.maybe_free_model_hooks()
            del self._pipeline
            self._pipeline = None

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

    async def generate(self, prompt: str, **params: Any) -> bytes:
        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        # Parse size string if present. Use assignment (not setdefault) so a
        # per-request `size` always wins over YAML `default_params.width/height`
        # — otherwise the request value is silently ignored when the YAML
        # already carries dimensions.
        if "size" in defaults:
            size = defaults.pop("size")
            if isinstance(size, str) and "x" in size:
                w, h = size.split("x")
                defaults["width"] = int(w)
                defaults["height"] = int(h)

        defaults.pop("response_format", None)
        defaults.pop("n", None)

        # Per-request scheduler override (applied inside the worker thread
        # so the swap + generate pair is atomic under queue.max_concurrent=1).
        scheduler_name = defaults.pop("scheduler", None)

        # Per-request LoRA adapters (applied *before* compel because LoRAs
        # can modify the text encoder; compel reads its live weights).
        loras = defaults.pop("loras", None)
        textual_inversions = defaults.pop("textual_inversions", None)
        highres_fix = defaults.pop("highres_fix", None)
        # `seed` is router-facing wire format; diffusers pipelines expect a
        # `generator` (torch.Generator) instead. Build it inside _gen() so
        # the device matches the pipeline's CUDA context at call time, and
        # so that a per-request seed doesn't leak across calls. Strict
        # pipelines (SD3, some FLUX variants) raise TypeError on stray
        # `seed` kwargs, which is why we pop it unconditionally.
        seed = defaults.pop("seed", None)
        # Decode base64 before the GPU executor hop so bad input 400s fast.
        image_b64 = defaults.pop("image", None)
        mask_b64 = defaults.pop("mask", None)
        denoising_strength = defaults.pop("denoising_strength", None)
        refiner_switch_at = defaults.pop("refiner_switch_at", None)
        if refiner_switch_at is not None and self._refiner is None:
            raise ValueError(
                "refiner_switch_at requires a refiner model — set "
                "model.refiner_hub_id in the YAML for this worker"
            )
        input_image = self._decode_image(image_b64, mode="RGB") if image_b64 else None
        input_mask = self._decode_image(mask_b64, mode="L") if mask_b64 else None
        if input_mask is not None and input_image is None:
            raise ValueError("mask requires image: inpainting needs a base image")
        model_dir = self._model_dir or "/app/models"
        lora_cfg = self.config.model.get("lora") or {}

        # Detect A1111-style prompt weighting — only activate compel path
        # when the syntax is actually used; plain prompts stay on the raw
        # tokenizer route so we don't subtly change baseline outputs.
        negative_prompt = defaults.get("negative_prompt") or ""
        use_compel = self._compel.available and has_weight_syntax(prompt, negative_prompt)

        def _gen():
            maybe_swap_scheduler(self._pipeline, scheduler_name)
            self._lora.apply(self._pipeline, loras, lora_cfg, model_dir)
            self._ti.apply(self._pipeline, textual_inversions, model_dir)
            if seed is not None:
                import torch
                gen_device = "cuda" if torch.cuda.is_available() else "cpu"
                defaults["generator"] = torch.Generator(device=gen_device).manual_seed(int(seed))
            if use_compel:
                self._compel.apply(prompt, negative_prompt or None, defaults)
            if highres_fix:
                return apply_highres_fix(
                    self._pipeline, self._ensure_img2img_pipe, prompt, defaults, highres_fix,
                )

            # Dispatch: inpaint (image+mask) → img2img (image) → text2img.
            if input_image is not None:
                call_kwargs = dict(defaults)
                if denoising_strength is not None:
                    call_kwargs["strength"] = float(denoising_strength)
                if input_mask is not None:
                    # Paint-app masks can come at any resolution — normalise.
                    from PIL import Image as _Image
                    mask = input_mask
                    if mask.size != input_image.size:
                        mask = mask.resize(input_image.size, _Image.NEAREST)
                    pipe = self._ensure_inpaint_pipe()
                    if use_compel:
                        return pipe(image=input_image, mask_image=mask, **call_kwargs).images[0]
                    return pipe(prompt=prompt, image=input_image, mask_image=mask, **call_kwargs).images[0]
                pipe = self._ensure_img2img_pipe()
                if use_compel:
                    return pipe(image=input_image, **call_kwargs).images[0]
                return pipe(prompt=prompt, image=input_image, **call_kwargs).images[0]

            # SDXL Refiner ensemble — base until `switch_at`, refiner finishes.
            if refiner_switch_at is not None:
                base_kwargs = dict(defaults)
                base_kwargs["denoising_end"] = float(refiner_switch_at)
                base_kwargs["output_type"] = "latent"
                if use_compel:
                    latents = self._pipeline(**base_kwargs).images
                else:
                    latents = self._pipeline(prompt=prompt, **base_kwargs).images

                # Refiner gets num_inference_steps, guidance_scale, generator;
                # size is driven by the latent shape so we drop width/height.
                ref_kwargs: dict[str, Any] = {
                    "image": latents,
                    "denoising_start": float(refiner_switch_at),
                }
                for k in ("num_inference_steps", "guidance_scale", "generator",
                          "negative_prompt"):
                    if k in defaults:
                        ref_kwargs[k] = defaults[k]
                # Refiner doesn't take compel-style prompt_embeds directly —
                # use the text prompt even when compel was on for base. The
                # refiner is polishing details, weighting is less critical.
                return self._refiner(prompt=prompt, **ref_kwargs).images[0]

            if use_compel:
                return self._pipeline(**defaults).images[0]
            # Pass `prompt` as kwarg — some pipelines (e.g. FLUX.2-klein)
            # have `image` as the first positional arg for img2img, so
            # positional `prompt` silently lands in the wrong slot.
            return self._pipeline(prompt=prompt, **defaults).images[0]

        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, _gen)

        # Memory hygiene: drop cached allocator blocks so back-to-back
        # requests don't accumulate fragmentation on tight VRAM budgets
        # (12GB cards running SDXL + compel + LoRA/TI registries are close
        # to the limit already). empty_cache is cheap and only releases
        # unused blocks — in-use tensors stay put.
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
