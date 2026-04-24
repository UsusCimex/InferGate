from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def apply_highres_fix(
    base_pipeline: Any,
    ensure_img2img_pipe: Callable[[], Any],
    prompt: str,
    defaults: dict[str, Any],
    hires: dict[str, Any],
) -> Any:
    """Two-pass high-res generation: compose at base resolution, upscale, img2img refine."""
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

    pass1_kwargs = dict(defaults)
    if "prompt_embeds" in pass1_kwargs:
        base_image = base_pipeline(**pass1_kwargs).images[0]
    else:
        base_image = base_pipeline(prompt=prompt, **pass1_kwargs).images[0]
    logger.info("HighresFix pass 1 complete: %dx%d", base_image.width, base_image.height)

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

    # Free base-pass allocations before the larger pass 2.
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    # Pass width/height explicitly — SDXL img2img auto-detect snaps to 1024-bucket.
    img2img_kwargs = {
        k: v for k, v in defaults.items()
        if k not in ("width", "height")
    }
    img2img_kwargs["width"] = new_w
    img2img_kwargs["height"] = new_h
    if hires_steps is not None:
        img2img_kwargs["num_inference_steps"] = int(hires_steps)

    img2img_pipe = ensure_img2img_pipe()
    logger.info(
        "HighresFix pass 2 (img2img): target=%dx%d strength=%.2f",
        new_w, new_h, denoising,
    )
    if "prompt_embeds" in img2img_kwargs:
        result = img2img_pipe(image=upscaled, strength=denoising, **img2img_kwargs).images[0]
    else:
        result = img2img_pipe(
            prompt=prompt, image=upscaled, strength=denoising, **img2img_kwargs
        ).images[0]
    logger.info("HighresFix pass 2 complete: %dx%d", result.width, result.height)
    return result
