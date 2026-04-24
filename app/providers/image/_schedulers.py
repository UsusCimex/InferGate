from __future__ import annotations

from typing import Any

# Values are either a diffusers class name or (class_name, extra_from_config_kwargs).
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
    """Replace `pipeline.scheduler` with the named alternative in-place."""
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
