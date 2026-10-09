"""CPU offload mode of a diffusion pipeline, `model.offload` in the model YAML."""
from __future__ import annotations

import os
import re
from typing import Any

OFFLOAD_MODES = ("none", "model", "sequential")
_LEGACY_KEYS = ("cpu_offload", "sequential_cpu_offload")


def offload_mode(model_id: str, model: dict[str, Any], allowed: tuple[str, ...] = OFFLOAD_MODES) -> str:
    """`none`, `model` or `sequential`; ValueError for an unknown mode or the old two flags."""
    prefix = re.sub(r"[^A-Za-z0-9]", "_", model_id).upper()
    legacy = [key for key in _LEGACY_KEYS if key in model]
    legacy += [name for name in (f"{prefix}_CPU_OFFLOAD", f"{prefix}_SEQUENTIAL_OFFLOAD") if name in os.environ]
    if legacy:
        raise ValueError(
            f"{model_id}: {', '.join(legacy)} no longer apply; "
            f"set offload in the YAML or {prefix}_OFFLOAD to one of {', '.join(allowed)}"
        )
    mode = str(model.get("offload") or "none").lower()
    if mode not in allowed:
        raise ValueError(f"{model_id}: offload must be one of {', '.join(allowed)}, got '{mode}'")
    return mode
