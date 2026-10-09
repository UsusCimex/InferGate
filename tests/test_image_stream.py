"""Image streaming over server-sent events and the plain PNG response of /v1/images/generations."""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest

from app.providers.base import ImageFrame, ImageProvider, WorkerStreamError
from app.routers import images as images_module
from tests.conftest import _make_model_config

_PNG = b"\x89PNG\r\n\x1a\n"


class _PreviewingProvider(ImageProvider):
    """Sends `partial_images` previews, then the final image; `fail_after` previews raise instead."""

    fail_after: int | None = None
    fail_with: Exception = RuntimeError("the GPU caught fire")

    async def load(self, model_dir: str) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def generate(self, prompt: str, **params: Any) -> bytes:
        return _PNG + b"final"

    async def generate_stream(self, prompt: str, partial_images: int, **params: Any):
        for i in range(partial_images):
            if self.fail_after == i:
                raise self.fail_with
            yield ImageFrame(_PNG + f"preview{i}".encode(), final=False)
        if self.fail_after == partial_images:
            raise self.fail_with
        yield ImageFrame(await self.generate(prompt, **params), final=True)


@pytest.fixture
def previewing(services):
    config = _make_model_config("test-previews", "image", "FakeImageProvider", "seed_only")
    provider = _PreviewingProvider(config)
    services["manager"]._registry["test-previews"] = provider
    services["scheduler"].register_model("test-previews", 1)
    return provider


def _events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


async def _settle() -> None:
    """Wait for streamed generations that finish after their response."""
    if images_module._STREAMS:
        await asyncio.wait(set(images_module._STREAMS))


@pytest.mark.asyncio
async def test_stream_sends_previews_then_the_final_image(client, previewing):
    resp = await client.post("/v1/images/generations", json={
        "model": "test-previews", "prompt": "a fox", "stream": True, "partial_images": 2,
    })

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-infergate-model"] == "test-previews"
    events = _events(resp.text)
    assert [name for name, _ in events] == [
        "image_generation.partial_image", "image_generation.partial_image", "image_generation.completed",
    ]
    assert [data.get("partial_image_index") for _, data in events] == [0, 1, None]
    assert [base64.b64decode(data["b64_json"]) for _, data in events] == [
        _PNG + b"preview0", _PNG + b"preview1", _PNG + b"final",
    ]
    assert all(data["type"] == name and data["output_format"] == "png" for name, data in events)


@pytest.mark.asyncio
async def test_a_provider_without_previews_streams_only_the_final_image(client):
    resp = await client.post("/v1/images/generations", json={
        "prompt": "a fox", "stream": True, "partial_images": 3,
    })

    [(name, data)] = _events(resp.text)
    assert name == "image_generation.completed"
    assert base64.b64decode(data["b64_json"]).startswith(_PNG)


@pytest.mark.asyncio
async def test_a_streamed_image_is_cached_and_a_repeat_comes_from_the_cache(client, previewing):
    request = {
        "model": "test-previews", "prompt": "a fox", "seed": 7, "stream": True, "partial_images": 1,
    }
    first = await client.post("/v1/images/generations", json=request)
    await _settle()
    repeat = await client.post("/v1/images/generations", json=request)

    assert first.headers["x-infergate-cache"] == "MISS"
    assert repeat.headers["x-infergate-cache"] == "HIT"
    [(name, data)] = _events(repeat.text)
    assert name == "image_generation.completed"
    assert base64.b64decode(data["b64_json"]) == _PNG + b"final"
    history = (await client.get("/ui/history")).json()["data"]
    assert [entry["cache"] for entry in history] == ["HIT", "MISS"]


@pytest.mark.asyncio
async def test_an_error_before_the_first_frame_keeps_its_status(client, previewing):
    previewing.fail_after = 0
    previewing.fail_with = ValueError("scheduler 'x' is not supported")

    resp = await client.post("/v1/images/generations", json={
        "model": "test-previews", "prompt": "a fox", "stream": True, "partial_images": 2,
    })

    assert resp.status_code == 400
    assert resp.json()["error"] == {"message": "scheduler 'x' is not supported", "type": "invalid_request"}


@pytest.mark.asyncio
async def test_a_broken_worker_stream_before_the_first_frame_is_a_502(client, previewing):
    previewing.fail_after = 0
    previewing.fail_with = WorkerStreamError("malformed line in the worker's image stream")

    resp = await client.post("/v1/images/generations", json={
        "model": "test-previews", "prompt": "a fox", "stream": True, "partial_images": 2,
    })

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(("error", "kind"), [
    (RuntimeError("the GPU caught fire"), "server_error"),
    (WorkerStreamError("the GPU caught fire"), "upstream_error"),
])
async def test_an_error_after_a_preview_ends_the_stream_with_an_error_event(
    client, previewing, error, kind
):
    previewing.fail_after = 1
    previewing.fail_with = error

    resp = await client.post("/v1/images/generations", json={
        "model": "test-previews", "prompt": "a fox", "stream": True, "partial_images": 2,
    })

    assert resp.status_code == 200
    events = _events(resp.text)
    assert [name for name, _ in events] == ["image_generation.partial_image", "error"]
    assert events[1][1] == {"error": {"message": "the GPU caught fire", "type": kind}}


@pytest.mark.asyncio
async def test_png_format_answers_with_the_image_itself(client):
    request = {"prompt": "a fox", "seed": 3, "response_format": "png"}
    resp = await client.post("/v1/images/generations", json=request)
    repeat = await client.post("/v1/images/generations", json=request)

    for answer, cache in ((resp, "MISS"), (repeat, "HIT")):
        assert answer.status_code == 200
        assert answer.headers["content-type"] == "image/png"
        assert answer.headers["x-infergate-cache"] == cache
        assert answer.content.startswith(_PNG)


@pytest.mark.asyncio
async def test_a_cache_hit_keeps_the_url_format(client):
    request = {"prompt": "a fox", "seed": 3, "response_format": "url"}
    await client.post("/v1/images/generations", json=request)
    repeat = await client.post("/v1/images/generations", json=request)

    assert repeat.headers["x-infergate-cache"] == "HIT"
    assert repeat.json()["data"][0]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [
    {"stream": True, "n": 2},
    {"response_format": "png", "n": 2},
    {"stream": True, "response_format": "url"},
    {"partial_images": 1},
    {"stream": True, "partial_images": 4},
    {"response_format": "jpeg"},
])
async def test_invalid_stream_and_format_combinations_are_rejected(client, extra):
    resp = await client.post("/v1/images/generations", json={"prompt": "a fox", **extra})
    assert resp.status_code == 422
