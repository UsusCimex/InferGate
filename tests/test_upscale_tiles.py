from __future__ import annotations

import io
import random
from types import SimpleNamespace

import pytest
from PIL import Image

from app.config import ModelConfig
from app.providers.upscale._tiles import upscale_tiled


def _noise(width: int, height: int) -> Image.Image:
    return Image.frombytes("RGB", (width, height), random.Random(width * height).randbytes(width * height * 3))


def _nearest(scale: int):
    def run(tile: Image.Image) -> Image.Image:
        return tile.resize((tile.width * scale, tile.height * scale), Image.NEAREST)
    return run


def test_tiles_stitch_into_the_whole_image_result():
    image = _noise(70, 45)
    out = upscale_tiled(image, 3, tile=16, pad=4, run=_nearest(3))
    assert out.size == (210, 135)
    assert out.tobytes() == _nearest(3)(image).tobytes()


def test_each_tile_runs_once_with_its_context():
    image, seen = _noise(70, 45), []

    def run(tile: Image.Image) -> Image.Image:
        seen.append(tile.size)
        return _nearest(2)(tile)

    upscale_tiled(image, 2, tile=16, pad=4, run=run)
    assert len(seen) == 5 * 3
    assert seen[0] == (20, 20)
    assert seen[6] == (24, 24)
    assert max(w for w, _ in seen) <= 16 + 2 * 4


def test_a_model_with_another_scale_is_refused():
    with pytest.raises(RuntimeError, match="should become"):
        upscale_tiled(_noise(40, 40), 4, tile=16, pad=4, run=_nearest(2))


def _provider(tile_size: int, max_input_side: int = 2048):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from app.providers.upscale.spandrel_provider import SpandrelUpscaleProvider

    config = ModelConfig(
        id="test-upscale", display_name="t", category="upscale",
        provider_class="SpandrelUpscaleProvider",
        model={"hub_id": "test/test", "filename": "x.pth", "tile_size": tile_size,
               "tile_pad": 4, "max_input_side": max_input_side},
    )
    provider = SpandrelUpscaleProvider(config)
    upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
    provider.calls = []

    def model(tensor):
        provider.calls.append(tuple(tensor.shape[-2:]))
        return upsample(tensor)

    provider._model = SimpleNamespace(model=model)
    provider._device, provider._dtype, provider._scale = "cpu", torch.float32, 2
    provider._loaded = True
    return provider


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize(("tile_size", "passes"), [(16, 5 * 3), (0, 1), (40, 1), (64, 1)])
async def test_provider_tiles_only_inputs_larger_than_a_tile(tile_size, passes):
    provider = _provider(tile_size)
    image = _noise(40, 24)
    out = Image.open(io.BytesIO(await provider.upscale(_png(image))))
    assert len(provider.calls) == passes
    assert out.tobytes() == _nearest(2)(image).tobytes()


async def test_no_tiled_pass_is_larger_than_tile_size():
    provider = _provider(16)
    await provider.upscale(_png(_noise(40, 24)))
    assert max(max(shape) for shape in provider.calls) == 16


async def test_a_tile_size_inside_its_padding_is_refused():
    provider = _provider(8)
    with pytest.raises(RuntimeError, match="leaves no image inside tile_pad"):
        await provider.upscale(_png(_noise(40, 24)))


async def test_provider_refuses_inputs_over_max_input_side():
    provider = _provider(16, max_input_side=32)
    with pytest.raises(ValueError, match="max_input_side=32"):
        await provider.upscale(_png(_noise(40, 24)))
