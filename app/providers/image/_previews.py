from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from app.providers.base import ImageFrame

logger = logging.getLogger(__name__)

_PREVIEW_MAX_SIDE = 512


def preview_steps(total: int, count: int) -> set[int]:
    """Step indexes, evenly spaced, after which a preview is sent; never the last step."""
    steps = {int(total * (i + 1) / (count + 1)) - 1 for i in range(count)}
    return {step for step in steps if 0 <= step < total - 1}


def latents_to_png(pipe: Any, latents: Any) -> bytes | None:
    """PNG of the latents of a running step through the pipeline's VAE.

    Only [batch, channels, height, width] latents decode this way; FLUX and Qwen-Image pack
    theirs differently and get None.
    """
    vae = getattr(pipe, "vae", None)
    processor = getattr(pipe, "image_processor", None)
    if latents is None or vae is None or processor is None or latents.ndim != 4:
        return None
    config = vae.config
    sample = latents[:1].to(vae.dtype) / config.scaling_factor
    sample = sample + (getattr(config, "shift_factor", None) or 0.0)
    decoded = vae.decode(sample, return_dict=False)[0]
    image = processor.postprocess(decoded, output_type="pil")[0]
    image.thumbnail((_PREVIEW_MAX_SIDE, _PREVIEW_MAX_SIDE))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class StepPreviews:
    """`callback_on_step_end` of a diffusers pipeline that hands PNG previews to `send`."""

    tensor_inputs = ("latents",)

    def __init__(
        self,
        count: int,
        send: Callable[[bytes], None],
        decode: Callable[[Any, Any], bytes | None] = latents_to_png,
    ) -> None:
        self._count = count
        self._send = send
        self._decode = decode
        self._steps: set[int] | None = None
        self._off = False

    def __call__(self, pipe: Any, step: int, _timestep: Any, callback_kwargs: dict) -> dict:
        if self._steps is None:
            self._steps = preview_steps(int(getattr(pipe, "num_timesteps", 0) or 0), self._count)
        if not self._off and step in self._steps:
            self._preview(pipe, callback_kwargs.get("latents"))
        return callback_kwargs

    def _preview(self, pipe: Any, latents: Any) -> None:
        try:
            png = self._decode(pipe, latents)
        except Exception as e:
            logger.warning("Step previews stop for this image: %s", e)
            png = None
        if png is None:
            self._off = True
            return
        self._send(png)


async def relay_previews(
    previews: asyncio.Queue[bytes], generation: asyncio.Future[bytes]
) -> AsyncIterator[ImageFrame]:
    """Previews as they arrive, then the result of `generation` as the final frame."""
    while True:
        waiter = asyncio.ensure_future(previews.get())
        try:
            await asyncio.wait({waiter, generation}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            waiter.cancel()
            raise
        if waiter.done():
            yield ImageFrame(waiter.result(), final=False)
            continue
        waiter.cancel()
        while not previews.empty():
            yield ImageFrame(previews.get_nowait(), final=False)
        yield ImageFrame(generation.result(), final=True)
        return
