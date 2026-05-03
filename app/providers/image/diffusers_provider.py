from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
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
    """No-op the warmup tensor preallocation that races with lazy CUDA init on Blackwell."""
    noop = lambda *_a, **_kw: None  # noqa: E731
    import importlib

    # `from X import Y` creates a local binding — every re-importing module must be patched.
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


def _apply_vae_tiling(pipe: Any, model_id: str, torch: Any) -> None:
    """Enable VAE tiling with the largest tile that fits in available VRAM.

    Measures free VRAM after the model is on device, then picks the biggest
    tile_latent_min_size where a single tile's decode fits comfortably.  When
    free VRAM is large enough for a full 128×128 decode the function enables
    tiling with threshold=129 (effectively a no-op for ≤1024px images).

    Empirical baseline: decoding a 64×64-latent (512×512px) tile consumes
    ~900 MB extra; usage scales quadratically with tile side length.
    """
    if not (torch.cuda.is_available() and hasattr(pipe, "enable_vae_tiling")):
        return

    # Half the free VRAM is reserved for UNet activations during inference.
    free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
    vae_budget_mb = free_mb * 0.5

    # Base: 64-latent tile ≈ 900 MB.  Scales as tile².
    _BASE_TILE = 64
    _BASE_MB = 900.0
    max_tile = int(_BASE_TILE * (vae_budget_mb / _BASE_MB) ** 0.5)

    if max_tile >= 128:
        # Full 1024px decode fits — set threshold above 128 to skip tiling.
        tile_size = 129
        logger.info(
            "VAE tiling: free VRAM %.0f MB, full-decode fits (tile_min=%d) for %s",
            free_mb, tile_size, model_id,
        )
    else:
        tile_size = max(48, max_tile)
        logger.info(
            "VAE tiling: free VRAM %.0f MB, budget %.0f MB → tile=%d latents for %s",
            free_mb, vae_budget_mb, tile_size, model_id,
        )

    pipe.enable_vae_tiling()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "tile_latent_min_size"):
        pipe.vae.tile_latent_min_size = tile_size


@register_provider
class DiffusersImageProvider(ImageProvider):
    """Image provider backed by any diffusers pipeline (FLUX, SD, PixArt, ...)."""

    def __init__(self, config):
        super().__init__(config)
        self._pipeline = None
        self._compel = CompelAdapter(self.model_id)
        self._lora = LoraCache(self.model_id)
        self._ti = TextualInversionRegistry(self.model_id)
        self._model_dir: str | None = None
        self._img2img_pipeline = None
        self._inpaint_pipeline = None
        self._refiner = None

    async def load(self, model_dir: str) -> None:
        import torch
        from diffusers import DiffusionPipeline

        self._model_dir = model_dir

        _disable_caching_allocator_warmup()

        # Blackwell (sm_120): eagerly init CUDA before first alloc to avoid cudaErrorNotReady.
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

        if self.config.model.get("drop_t5", False):
            kwargs["text_encoder_3"] = None
            kwargs["tokenizer_3"] = None

        quantization = self.config.model.get("quantization")
        if quantization:
            kwargs["quantization_config"] = self._build_quantization_config(
                quantization, dtype
            )

        cpu_offload = self.config.model.get("cpu_offload", False)
        sequential_offload = self.config.model.get("sequential_cpu_offload", False)

        logger.info("Loading %s from %s", self.model_id, hub_id)
        loop = asyncio.get_running_loop()

        vae_tiling = self.config.model.get("vae_tiling", False)

        def _load():
            import torch as _torch

            pipe = DiffusionPipeline.from_pretrained(hub_id, **kwargs)
            if sequential_offload:
                pipe.enable_sequential_cpu_offload()
            elif cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            if vae_tiling and hasattr(pipe, "enable_vae_tiling"):
                _apply_vae_tiling(pipe, self.model_id, _torch)
            return pipe

        self._pipeline = await loop.run_in_executor(_GPU_EXECUTOR, _load)
        if self.config.model.get("compel", True):
            self._compel.init(self._pipeline)

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

        if self.config.model.get("warmup", True):
            logger.info("Warming up %s …", self.model_id)
            await loop.run_in_executor(_GPU_EXECUTOR, self._warmup)
            logger.info("Warmup complete for %s", self.model_id)

        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    def _ensure_img2img_pipe(self):
        """Return the img2img pipeline, building it lazily from shared base weights."""
        from diffusers import AutoPipelineForImage2Image

        if self._img2img_pipeline is None:
            self._img2img_pipeline = AutoPipelineForImage2Image.from_pipe(self._pipeline)
        # Re-sync scheduler so per-request swap on base applies here too.
        self._img2img_pipeline.scheduler = self._pipeline.scheduler
        return self._img2img_pipeline

    def _ensure_inpaint_pipe(self):
        """Return the inpaint pipeline, building it lazily from shared base weights."""
        from diffusers import AutoPipelineForInpainting

        if self._inpaint_pipeline is None:
            self._inpaint_pipeline = AutoPipelineForInpainting.from_pipe(self._pipeline)
        self._inpaint_pipeline.scheduler = self._pipeline.scheduler
        return self._inpaint_pipeline

    @staticmethod
    def _decode_image(b64_str: str, mode: str | None = None):
        """Decode base64 (with optional data: prefix) into a PIL.Image."""
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
        """Run one minimal inference to trigger kernel autotune and offload hooks."""
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
        """Build a diffusers PipelineQuantizationConfig from the YAML quantization name."""
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
                # If a prior CUDA OOM left the context broken, each call may raise;
                # proceed anyway so the provider is marked unloaded and state stays consistent.
                for fn in (torch.cuda.synchronize, torch.cuda.empty_cache, torch.cuda.ipc_collect):
                    with contextlib.suppress(Exception):
                        fn()
            await loop.run_in_executor(None, _cuda_cleanup)
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def generate(self, prompt: str, **params: Any) -> bytes:
        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        # Per-request `size` must win over YAML width/height; use assignment, not setdefault.
        if "size" in defaults:
            size = defaults.pop("size")
            if isinstance(size, str) and "x" in size:
                w, h = size.split("x")
                defaults["width"] = int(w)
                defaults["height"] = int(h)

        defaults.pop("response_format", None)
        defaults.pop("n", None)

        scheduler_name = defaults.pop("scheduler", None)
        # LoRAs apply before compel: compel reads live text-encoder weights.
        loras = defaults.pop("loras", None)
        textual_inversions = defaults.pop("textual_inversions", None)
        highres_fix = defaults.pop("highres_fix", None)
        # `seed` is wire-format only — diffusers wants a torch.Generator. Build inside
        # _gen() so the device matches CUDA context, and pop unconditionally because
        # strict pipelines (SD3, some FLUX variants) raise TypeError on stray `seed`.
        seed = defaults.pop("seed", None)
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

            if input_image is not None:
                call_kwargs = dict(defaults)
                if denoising_strength is not None:
                    call_kwargs["strength"] = float(denoising_strength)
                if input_mask is not None:
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

            if refiner_switch_at is not None:
                base_kwargs = dict(defaults)
                base_kwargs["denoising_end"] = float(refiner_switch_at)
                base_kwargs["output_type"] = "latent"
                if use_compel:
                    latents = self._pipeline(**base_kwargs).images
                else:
                    latents = self._pipeline(prompt=prompt, **base_kwargs).images

                ref_kwargs: dict[str, Any] = {
                    "image": latents,
                    "denoising_start": float(refiner_switch_at),
                }
                for k in ("num_inference_steps", "guidance_scale", "generator",
                          "negative_prompt"):
                    if k in defaults:
                        ref_kwargs[k] = defaults[k]
                # Refiner uses the plain prompt; compel prompt_embeds aren't supported here.
                return self._refiner(prompt=prompt, **ref_kwargs).images[0]

            if use_compel:
                return self._pipeline(**defaults).images[0]
            # Pass prompt as kwarg: some pipelines (FLUX.2-klein) take `image` first positionally.
            return self._pipeline(prompt=prompt, **defaults).images[0]

        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, _gen)

        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
