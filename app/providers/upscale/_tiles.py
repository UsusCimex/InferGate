from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def upscale_tiled(
    image: Image.Image,
    scale: int,
    tile: int,
    pad: int,
    run: Callable[[Image.Image], Image.Image],
) -> Image.Image:
    """Upscale `image` tile by tile; each tile carries `pad` px of context that is cut from its result."""
    from PIL import Image as PILImage

    width, height = image.size
    out = PILImage.new("RGB", (width * scale, height * scale))
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            right, bottom = min(left + tile, width), min(top + tile, height)
            box = (max(left - pad, 0), max(top - pad, 0), min(right + pad, width), min(bottom + pad, height))
            result = run(image.crop(box))
            expected = ((box[2] - box[0]) * scale, (box[3] - box[1]) * scale)
            if result.size != expected:
                raise RuntimeError(f"model returned {result.size} for a tile that should become {expected}")
            inner = (
                (left - box[0]) * scale, (top - box[1]) * scale,
                (right - box[0]) * scale, (bottom - box[1]) * scale,
            )
            out.paste(result.crop(inner), (left * scale, top * scale))
    return out
