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


def resolve_scheduler(default: Any, name: str | None) -> Any:
    """Per-request scheduler: `default`, or the named one built from `default`'s config."""
    if not name:
        return default
    if type(default).__name__.startswith("FlowMatch"):
        raise ValueError(
            f"scheduler '{name}' does not apply to flow-matching models; "
            "omit it to keep the model's own sampler"
        )
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
    return cls.from_config(default.config, **extra)
