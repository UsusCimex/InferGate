from __future__ import annotations

from typing import Any

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


def maybe_swap_scheduler(pipeline: Any, name: str | None) -> None:
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
