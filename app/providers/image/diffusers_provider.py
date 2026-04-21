from __future__ import annotations

import asyncio
import concurrent.futures
import io
import itertools
import logging
import re
import threading
from collections import OrderedDict
from typing import Any

from app.providers.base import ImageProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)

# A1111-style weight syntax: (word:1.5) / (word, phrase:0.8) / (word:-1.2)
# Simple regex — catches the common form without trying to parse the full
# compel grammar. If this matches, we route the prompt through compel;
# otherwise we pass the raw string to the pipeline (cheaper, no semantic shift).
_WEIGHT_RE = re.compile(r"\([^()]+:\s*[-+]?\d+\.?\d*\s*\)")

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
        # diffusers
        ("diffusers.models.model_loading_utils", "_caching_allocator_warmup"),
        ("diffusers.models.modeling_utils", "_caching_allocator_warmup"),
        # transformers
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


# Per-request scheduler override. Values are either a diffusers class name
# or a (class_name, extra_kwargs) tuple passed to `Cls.from_config(...)`.
# All UNet/DiT/MMDiT pipelines accept these — Flow-Matching pipelines
# (FLUX, SD3.x) have their own FlowMatchEulerDiscreteScheduler and will
# fail a swap; the caller is expected to match scheduler class to model.
_SCHEDULERS: dict[str, str | tuple[str, dict[str, Any]]] = {
    "euler":            "EulerDiscreteScheduler",
    "euler_a":          "EulerAncestralDiscreteScheduler",
    "euler_ancestral":  "EulerAncestralDiscreteScheduler",
    "dpm++_2m":         "DPMSolverMultistepScheduler",
    "dpm++_2m_karras":  ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "dpm++_sde":        "DPMSolverSDEScheduler",
    "ddim":             "DDIMScheduler",
    "ddpm":             "DDPMScheduler",
    "lms":              "LMSDiscreteScheduler",
    "heun":             "HeunDiscreteScheduler",
    "pndm":             "PNDMScheduler",
    "unipc":            "UniPCMultistepScheduler",
}


def _maybe_swap_scheduler(pipeline: Any, name: str | None) -> None:
    """Replace `pipeline.scheduler` with a named alternative, in-place.

    Safe because every worker runs queue.max_concurrent=1 — no other thread
    is mid-generate on the same pipeline. No-op when `name` is None/empty.
    """
    if not name:
        return
    entry = _SCHEDULERS.get(name.lower())
    if entry is None:
        raise ValueError(
            f"Unknown scheduler '{name}'. Known: {sorted(_SCHEDULERS)}"
        )
    cls_name, extra = (entry if isinstance(entry, tuple) else (entry, {}))
    import diffusers

    try:
        cls = getattr(diffusers, cls_name)
    except AttributeError as e:
        raise ValueError(
            f"Scheduler class '{cls_name}' not in current diffusers version"
        ) from e
    pipeline.scheduler = cls.from_config(pipeline.scheduler.config, **extra)


@register_provider
class DiffusersImageProvider(ImageProvider):
    """Universal provider for any diffusers-compatible model.
    Supports FLUX, Stable Diffusion, PixArt, etc.
    The specific model is determined by the YAML config (hub_id).
    """

    def __init__(self, config):
        super().__init__(config)
        self._pipeline = None
        self._compel = None          # type: ignore[assignment]
        self._compel_mode = None     # "sdxl" (dual-encoder) | "sd15" (single) | None

        # LoRA adapter cache: maps (repo_id, weight_file) → adapter_name
        # registered in the pipeline. OrderedDict doubles as an LRU — move
        # to end on access, evict from front when size exceeds cap.
        self._lora_cache: OrderedDict[tuple[str, str | None], str] = OrderedDict()
        self._lora_counter = itertools.count()
        self._lora_lock = threading.Lock()
        self._model_dir: str | None = None  # stored at load() for LoRA downloads

        # Textual Inversion dedup set: (repo_id, weight_file, token). Unlike
        # LoRA, TIs can't be "deactivated" — once registered in the tokenizer
        # they persist for the lifetime of the pipeline (but are only visible
        # via their token in the prompt).
        self._ti_loaded: set[tuple[str, str | None, str | None]] = set()
        self._ti_lock = threading.Lock()

        # Img2img pipeline for HighresFix. Built lazily via `from_pipe` so it
        # shares UNet / VAE / text encoders with the base pipeline (zero VRAM
        # overhead). None until first highres_fix request touches it.
        self._img2img_pipeline = None

    async def load(self, model_dir: str) -> None:
        import torch
        from diffusers import DiffusionPipeline

        self._model_dir = model_dir  # reused by LoRA downloads in generate()

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
        self._init_compel()

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

    def _init_compel(self) -> None:
        """Attempt to initialise a Compel encoder for A1111-style weighting.

        Only works on CLIP-based pipelines that consume `prompt_embeds` in
        the SDXL/SD1.5 shape. Pipelines with T5-mixed embeds (FLUX, SD3) or
        non-CLIP encoders (Qwen-Image, Hunyuan-DiT) are explicitly skipped:
          * FLUX's `text_encoder_2` is T5 — Compel can't encode it.
          * SD3Pipeline concatenates CLIP(L+G)+T5 into [B,154,4096]; Compel's
            SDXL path emits [B,77,2048] which is a runtime shape mismatch
            even when T5 is dropped (pipeline still pads CLIP to 4096).

        Detection is by class name rather than duck-typing: SD3 exposes the
        same `tokenizer_2 + text_encoder_2` pair as SDXL (both are CLIP-G),
        so the capability test alone can't distinguish them.

        Controlled by YAML `compel: true|false` (default true).
        """
        if not self.config.model.get("compel", True):
            return
        try:
            from compel import Compel, ReturnedEmbeddingsType
        except ImportError:
            logger.info("compel not installed; prompt weighting unavailable for %s", self.model_id)
            return

        pipe = self._pipeline
        pipe_class = type(pipe).__name__
        # Hard-coded block list for architectures known to reject SDXL-shape
        # embeds. Matches diffusers class names (StableDiffusion3Pipeline,
        # StableDiffusion3Img2ImgPipeline, FluxPipeline, FluxImg2ImgPipeline).
        if "StableDiffusion3" in pipe_class or pipe_class.startswith("Flux"):
            logger.info(
                "Compel skipped for %s (%s): architecture uses T5-mixed embeds "
                "that are incompatible with compel's SDXL output shape",
                self.model_id, pipe_class,
            )
            return

        try:
            if (
                hasattr(pipe, "tokenizer_2")
                and hasattr(pipe, "text_encoder_2")
                and pipe.tokenizer_2 is not None
                and pipe.text_encoder_2 is not None
            ):
                # SDXL-style dual-encoder, penultimate hidden states + pooled output on enc-2.
                self._compel = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                )
                self._compel_mode = "sdxl"
                logger.info("Compel initialised for %s (sdxl dual-encoder)", self.model_id)
            elif (
                hasattr(pipe, "tokenizer")
                and hasattr(pipe, "text_encoder")
                and pipe.tokenizer is not None
                and pipe.text_encoder is not None
            ):
                self._compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
                self._compel_mode = "sd15"
                logger.info("Compel initialised for %s (single-encoder)", self.model_id)
            else:
                logger.info("Compel skipped for %s (no compatible tokenizer/text_encoder)", self.model_id)
        except Exception as e:  # noqa: BLE001 — init can fail on odd pipeline layouts
            logger.warning("Compel init failed for %s: %s — weighting disabled", self.model_id, e)
            self._compel = None
            self._compel_mode = None

    def _apply_loras(self, loras: list[dict] | None, model_dir: str) -> None:
        """Ensure requested LoRAs are loaded in the pipeline and activated.

        `loras` is a list of dicts with keys {id, weight, weight_file,
        adapter_name}. When None or empty we deactivate all adapters
        without evicting them (next matching request is still fast).

        Cache strategy:
          - Each (repo_id, weight_file) pair maps to one adapter_name in
            the pipeline; reused across requests.
          - OrderedDict LRU; when size exceeds `model.lora.max_loaded`
            (default 8) the front entry is evicted via delete_adapters.
          - Weight is per-request state — set via set_adapters, never
            baked into the adapter itself.
        """
        pipe = self._pipeline
        if pipe is None:
            return

        # Deactivate path: no loras requested for this call.
        if not loras:
            if hasattr(pipe, "disable_lora"):
                pipe.disable_lora()
            return

        lora_cfg = self.config.model.get("lora") or {}
        max_loaded = int(lora_cfg.get("max_loaded", 8))
        max_per_request = int(lora_cfg.get("max_per_request", 5))
        if len(loras) > max_per_request:
            raise ValueError(
                f"loras: {len(loras)} adapters requested, max {max_per_request}"
            )

        if not hasattr(pipe, "load_lora_weights"):
            raise ValueError(
                f"Pipeline for {self.model_id} does not support LoRA loading"
            )

        with self._lora_lock:
            active_names: list[str] = []
            active_weights: list[float] = []

            for spec in loras:
                repo_id = spec["id"]
                weight_file = spec.get("weight_file")
                cache_key = (repo_id, weight_file)

                if cache_key in self._lora_cache:
                    adapter_name = self._lora_cache[cache_key]
                    self._lora_cache.move_to_end(cache_key)
                    logger.debug("LoRA cache hit: %s → %s", repo_id, adapter_name)
                else:
                    # Client may pin an adapter_name; otherwise we allocate
                    # `lora_N` monotonically. Collisions (same name, diff
                    # repo) rename to a fresh slot to stay unambiguous.
                    adapter_name = spec.get("adapter_name") or f"lora_{next(self._lora_counter)}"
                    while adapter_name in self._lora_cache.values():
                        adapter_name = f"lora_{next(self._lora_counter)}"

                    load_kwargs: dict[str, Any] = {
                        "cache_dir": model_dir,
                        "adapter_name": adapter_name,
                    }
                    if weight_file:
                        load_kwargs["weight_name"] = weight_file

                    try:
                        logger.info(
                            "Loading LoRA %s%s into %s as '%s'",
                            repo_id,
                            f" (file={weight_file})" if weight_file else "",
                            self.model_id,
                            adapter_name,
                        )
                        pipe.load_lora_weights(repo_id, **load_kwargs)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to load LoRA '{repo_id}'"
                            f"{f' (file={weight_file})' if weight_file else ''}: {e}"
                        ) from e

                    self._lora_cache[cache_key] = adapter_name

                    # LRU eviction after insert, not before — a brand-new
                    # request can overshoot by exactly one and then be trimmed.
                    while len(self._lora_cache) > max_loaded:
                        evict_key, evict_name = self._lora_cache.popitem(last=False)
                        logger.info(
                            "Evicting LoRA adapter '%s' (%s) — cache full (%d)",
                            evict_name, evict_key[0], max_loaded,
                        )
                        try:
                            pipe.delete_adapters([evict_name])
                        except Exception as e:  # noqa: BLE001 — eviction is best-effort
                            logger.warning("delete_adapters(%s) failed: %s", evict_name, e)

                active_names.append(adapter_name)
                active_weights.append(float(spec.get("weight", 1.0)))

            # enable_lora is a no-op if LoRA state is already enabled but
            # cheap — call unconditionally for defensive-coding reasons.
            if hasattr(pipe, "enable_lora"):
                pipe.enable_lora()
            pipe.set_adapters(active_names, adapter_weights=active_weights)
            logger.debug(
                "Active LoRAs for request: %s",
                list(zip(active_names, active_weights, strict=True)),
            )

    def _apply_textual_inversions(self, tis: list[dict] | None, model_dir: str) -> None:
        """Register any not-yet-seen Textual Inversion embeddings with the
        pipeline. No-op on cache hit; no deactivation concept.

        A TI is identified by (repo_id, weight_file, token). Once registered
        the `<token>` string becomes usable in any prompt for the lifetime
        of the provider — no per-request state to apply.
        """
        if not tis:
            return
        pipe = self._pipeline
        if pipe is None:
            return
        if not hasattr(pipe, "load_textual_inversion"):
            raise ValueError(
                f"Pipeline for {self.model_id} does not support textual inversions"
            )

        with self._ti_lock:
            for spec in tis:
                repo_id = spec["id"]
                weight_file = spec.get("weight_file")
                token = spec.get("token")
                # Lists aren't hashable — normalise to a tuple for the set key.
                # The value we pass to load_textual_inversion keeps its original
                # type (diffusers accepts both str and list).
                token_key: Any = tuple(token) if isinstance(token, list) else token
                cache_key = (repo_id, weight_file, token_key)

                if cache_key in self._ti_loaded:
                    logger.debug("TI cache hit: %s (token=%s)", repo_id, token)
                    continue

                load_kwargs: dict[str, Any] = {"cache_dir": model_dir}
                if weight_file:
                    load_kwargs["weight_name"] = weight_file
                if token:
                    load_kwargs["token"] = token

                try:
                    logger.info(
                        "Registering textual inversion %s%s%s into %s",
                        repo_id,
                        f" (file={weight_file})" if weight_file else "",
                        f" (token={token})" if token else "",
                        self.model_id,
                    )
                    try:
                        pipe.load_textual_inversion(repo_id, **load_kwargs)
                    except Exception as single_call_err:
                        # SDXL "pivotal" TIs store separate clip_l / clip_g
                        # tensors in one file. diffusers' single-call path
                        # rejects that layout ("Loaded state dictionary is
                        # incorrect"); fall back to explicit per-encoder
                        # loading via safetensors + two load_textual_inversion
                        # calls.
                        if (
                            "clip_l" in str(single_call_err)
                            and "clip_g" in str(single_call_err)
                            and weight_file
                            and hasattr(pipe, "text_encoder_2")
                            and hasattr(pipe, "tokenizer_2")
                        ):
                            logger.info(
                                "Single-call load rejected dual-tensor TI; "
                                "retrying with per-encoder pivotal loading"
                            )
                            self._load_pivotal_ti(repo_id, weight_file, token, model_dir)
                        else:
                            raise
                except Exception as e:
                    raise ValueError(
                        f"Failed to load textual inversion '{repo_id}'"
                        f"{f' (file={weight_file})' if weight_file else ''}"
                        f"{f' (token={token})' if token else ''}: {e}"
                    ) from e

                self._ti_loaded.add(cache_key)

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
        # Re-point scheduler to the base pipeline's current one — cheap and
        # tolerates a per-request swap that happened moments ago.
        self._img2img_pipeline.scheduler = self._pipeline.scheduler
        return self._img2img_pipeline

    def _apply_highres_fix(self, prompt: str, defaults: dict, hires: dict):
        """Two-pass generation — compose at base res, upscale, img2img refine.

        Inputs:
          prompt      — positive prompt (compel embeds live in `defaults` if active)
          defaults    — pipeline kwargs for pass 1 (mutated in-place: width/height
                        become the BASE resolution; scheduler/size keys are already popped)
          hires       — {scale, denoising_strength, steps?, upscaler}

        Returns: PIL image at (width*scale, height*scale).
        """
        from PIL import Image

        scale = float(hires.get("scale", 2.0))
        denoising = float(hires.get("denoising_strength", 0.5))
        hires_steps = hires.get("steps")
        upscaler = str(hires.get("upscaler", "lanczos")).lower()

        base_w = int(defaults.get("width", 1024))
        base_h = int(defaults.get("height", 1024))
        logger.info(
            "HighresFix pass 1: base=%dx%d scale=%.2f denoising=%.2f upscaler=%s",
            base_w, base_h, scale, denoising, upscaler,
        )

        # Pass 1: generate at base resolution. `prompt` might be None when
        # compel supplied prompt_embeds — respect that.
        pass1_kwargs = dict(defaults)
        if "prompt_embeds" in pass1_kwargs:
            base_image = self._pipeline(**pass1_kwargs).images[0]
        else:
            base_image = self._pipeline(prompt=prompt, **pass1_kwargs).images[0]
        logger.info("HighresFix pass 1 complete: %dx%d", base_image.width, base_image.height)

        # Upscale (CPU / PIL — cheap, keeps VRAM clean for the img2img pass).
        new_w = int(base_w * scale)
        new_h = int(base_h * scale)
        resampler = {
            "nearest":  Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic":  Image.BICUBIC,
            "lanczos":  Image.LANCZOS,
        }.get(upscaler, Image.LANCZOS)
        upscaled = base_image.resize((new_w, new_h), resampler)
        logger.info("HighresFix upscaled: %dx%d", upscaled.width, upscaled.height)

        # Free base-pass intermediate allocations before the (larger) pass 2.
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        # Pass 2: img2img refine on the upscaled image. Pass width/height
        # EXPLICITLY — SDXL img2img's auto-detect from input image isn't
        # reliable (observed snapping to 1024 bucket). Better to be explicit.
        img2img_kwargs = {
            k: v for k, v in defaults.items()
            if k not in ("width", "height")
        }
        img2img_kwargs["width"] = new_w
        img2img_kwargs["height"] = new_h
        if hires_steps is not None:
            img2img_kwargs["num_inference_steps"] = int(hires_steps)

        img2img_pipe = self._ensure_img2img_pipe()
        logger.info("HighresFix pass 2 (img2img): target=%dx%d strength=%.2f", new_w, new_h, denoising)
        if "prompt_embeds" in img2img_kwargs:
            result = img2img_pipe(image=upscaled, strength=denoising, **img2img_kwargs).images[0]
        else:
            result = img2img_pipe(
                prompt=prompt, image=upscaled, strength=denoising, **img2img_kwargs
            ).images[0]
        logger.info("HighresFix pass 2 complete: %dx%d", result.width, result.height)
        return result

    def _load_pivotal_ti(
        self,
        repo_id: str,
        weight_file: str,
        token: Any,
        model_dir: str,
    ) -> None:
        """Load an SDXL pivotal TI (single file, separate clip_l / clip_g tensors).

        The .safetensors file is fetched via hf_hub_download, parsed into a
        dict of tensors, then each key is registered against the matching
        (tokenizer, text_encoder) pair of the dual-encoder pipeline.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        pipe = self._pipeline
        local_path = hf_hub_download(
            repo_id=repo_id, filename=weight_file, cache_dir=model_dir
        )
        sd = load_file(local_path)
        if "clip_l" not in sd or "clip_g" not in sd:
            raise ValueError(
                f"Pivotal TI fallback expected tensors 'clip_l' and 'clip_g' "
                f"in {weight_file}, got {list(sd.keys())}"
            )
        pipe.load_textual_inversion(
            sd["clip_l"],
            token=token,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
        )
        pipe.load_textual_inversion(
            sd["clip_g"],
            token=token,
            text_encoder=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer_2,
        )

    def _apply_compel(self, prompt: str, negative_prompt: str | None, defaults: dict) -> None:
        """Replace `prompt`/`negative_prompt` in `defaults` with compel embeds.

        Removes any conflicting string-prompt keys so the pipeline's
        mutual-exclusion checks don't reject the call.
        """
        defaults.pop("negative_prompt", None)
        if self._compel_mode == "sdxl":
            p_embeds, p_pooled = self._compel(prompt)
            defaults["prompt_embeds"] = p_embeds
            defaults["pooled_prompt_embeds"] = p_pooled
            if negative_prompt:
                n_embeds, n_pooled = self._compel(negative_prompt)
                defaults["negative_prompt_embeds"] = n_embeds
                defaults["negative_pooled_prompt_embeds"] = n_pooled
        else:  # sd15
            defaults["prompt_embeds"] = self._compel(prompt)
            if negative_prompt:
                defaults["negative_prompt_embeds"] = self._compel(negative_prompt)

    def _warmup(self) -> None:
        """Minimal dummy inference to trigger kernel autotuning + offload hooks."""
        defaults = dict(self.config.model.get("default_params", {}))
        # Override to the cheapest possible run: 1 step, 256×256.
        defaults.pop("response_format", None)
        defaults.pop("n", None)
        defaults.pop("size", None)
        defaults["num_inference_steps"] = 1
        defaults["width"] = 256
        defaults["height"] = 256
        try:
            self._pipeline(prompt="warmup", **defaults)
        except Exception as e:  # noqa: BLE001 — warmup failure shouldn't block load
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

        # Remove params not accepted by pipeline
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
        model_dir = self._model_dir or "/app/models"

        # Detect A1111-style prompt weighting — only activate compel path
        # when the syntax is actually used; plain prompts stay on the raw
        # tokenizer route so we don't subtly change baseline outputs.
        negative_prompt = defaults.get("negative_prompt") or ""
        use_compel = self._compel is not None and (
            _WEIGHT_RE.search(prompt) or _WEIGHT_RE.search(negative_prompt)
        )

        def _gen():
            _maybe_swap_scheduler(self._pipeline, scheduler_name)
            self._apply_loras(loras, model_dir)
            self._apply_textual_inversions(textual_inversions, model_dir)
            if use_compel:
                self._apply_compel(prompt, negative_prompt or None, defaults)
            if highres_fix:
                return self._apply_highres_fix(prompt, defaults, highres_fix)
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
