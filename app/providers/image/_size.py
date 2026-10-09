"""The request's `size` over the YAML width and height."""
from __future__ import annotations

from typing import Any


def apply_size(params: dict[str, Any]) -> None:
    """Replace a `WxH` size in `params` with width and height, overriding the YAML defaults."""
    size = params.pop("size", None)
    if isinstance(size, str) and "x" in size:
        width, height = size.split("x")
        params["width"] = int(width)
        params["height"] = int(height)
