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

        Only works on CLIP-based pipelines. Dual-encoder pipelines (SDXL
        family) need requires_pooled for the second tokenizer. Everything
        T5/Qwen-VL/mT5-based (FLUX, SD3 with T5, Qwen-Image, Hunyuan-DiT)
        is not CLIP — compel init fails gracefully and we fall back to
        raw-prompt mode.

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

        # Parse size string if present
        if "size" in defaults:
            size = defaults.pop("size")
            if isinstance(size, str) and "x" in size:
                w, h = size.split("x")
                defaults.setdefault("width", int(w))
                defaults.setdefault("height", int(h))

        # Remove params not accepted by pipeline
        defaults.pop("response_format", None)
        defaults.pop("n", None)

        # Per-request scheduler override (applied inside the worker thread
        # so the swap + generate pair is atomic under queue.max_concurrent=1).
        scheduler_name = defaults.pop("scheduler", None)

        # Per-request LoRA adapters (applied *before* compel because LoRAs
        # can modify the text encoder; compel reads its live weights).
        loras = defaults.pop("loras", None)
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
            if use_compel:
                self._apply_compel(prompt, negative_prompt or None, defaults)
                return self._pipeline(**defaults).images[0]
            # Pass `prompt` as kwarg — some pipelines (e.g. FLUX.2-klein)
            # have `image` as the first positional arg for img2img, so
            # positional `prompt` silently lands in the wrong slot.
            return self._pipeline(prompt=prompt, **defaults).images[0]

        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, _gen)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
