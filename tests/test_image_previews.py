"""Step previews of diffusers pipelines, without torch: schedule, callback and relay."""
from __future__ import annotations

import asyncio
import logging

import pytest

from app.providers.base import ImageFrame
from app.providers.image._previews import StepPreviews, preview_steps, relay_previews


class _Pipe:
    num_timesteps = 4


@pytest.mark.parametrize(("total", "count", "steps"), [
    (30, 3, {6, 14, 21}),
    (4, 3, {0, 1, 2}),
    (4, 1, {1}),
    (1, 3, set()),
    (0, 2, set()),
])
def test_preview_steps_spread_over_the_run_and_skip_the_last(total, count, steps):
    assert preview_steps(total, count) == steps


def test_step_previews_decode_the_chosen_steps():
    sent: list[bytes] = []
    callback = StepPreviews(2, sent.append, lambda pipe, latents: f"png{latents}".encode())

    for step in range(4):
        kwargs = {"latents": step}
        assert callback(_Pipe(), step, None, kwargs) is kwargs

    assert sent == [b"png0", b"png1"]


def test_a_failed_decode_stops_the_previews_of_this_image(caplog):
    calls: list[int] = []

    def decode(pipe, latents):
        calls.append(latents)
        raise RuntimeError("out of memory")

    sent: list[bytes] = []
    callback = StepPreviews(3, sent.append, decode)
    with caplog.at_level(logging.WARNING, logger="app.providers.image._previews"):
        for step in range(4):
            callback(_Pipe(), step, None, {"latents": step})

    assert calls == [0]
    assert sent == []
    assert "out of memory" in caplog.text


def test_latents_without_a_preview_stop_asking():
    calls: list[int] = []
    callback = StepPreviews(3, [].append, lambda pipe, latents: calls.append(latents))

    for step in range(4):
        callback(_Pipe(), step, None, {"latents": step})

    assert calls == [0]


@pytest.mark.asyncio
async def test_previews_from_the_pipeline_thread_come_before_the_final_image():
    loop = asyncio.get_running_loop()
    previews: asyncio.Queue[bytes] = asyncio.Queue()

    def run_pipeline() -> bytes:
        for i in range(3):
            loop.call_soon_threadsafe(previews.put_nowait, f"p{i}".encode())
        return b"final"

    generation = loop.run_in_executor(None, run_pipeline)
    frames = [frame async for frame in relay_previews(previews, generation)]

    assert frames == [
        ImageFrame(b"p0", final=False),
        ImageFrame(b"p1", final=False),
        ImageFrame(b"p2", final=False),
        ImageFrame(b"final", final=True),
    ]


@pytest.mark.asyncio
async def test_relay_raises_the_error_of_the_generation():
    generation = asyncio.get_running_loop().create_future()
    generation.set_exception(ValueError("size must be WxH"))

    with pytest.raises(ValueError, match="size must be WxH"):
        [frame async for frame in relay_previews(asyncio.Queue(), generation)]
