"""AuraSrUpscaleProvider against a fake aura_sr module: local weights, 4x PNG, a fixed seed, the input limit."""
from __future__ import annotations

import io
import sys
import types
from typing import ClassVar

import pytest
from PIL import Image

from app.config import ModelConfig
from app.providers.upscale.aurasr_provider import AuraSrUpscaleProvider


class _FakeAuraSr:
    loaded_from: ClassVar[list[str]] = []

    @classmethod
    def from_pretrained(cls, model_id):
        cls.loaded_from.append(model_id)
        return cls()

    def upscale_4x_overlapped(self, image):
        return image.resize((image.width * 4, image.height * 4), Image.NEAREST)


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    seeds = []
    monkeypatch.setitem(sys.modules, "aura_sr", types.SimpleNamespace(AuraSR=_FakeAuraSr))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        manual_seed=seeds.append, cuda=types.SimpleNamespace(is_available=lambda: False),
    ))
    snapshot = tmp_path / "snapshot"
    hub = types.SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    _FakeAuraSr.loaded_from = []
    return seeds, snapshot


def _provider(max_input_side: int = 2048) -> AuraSrUpscaleProvider:
    return AuraSrUpscaleProvider(ModelConfig(
        id="aurasr-v2", display_name="AuraSR v2", category="upscale", provider_class="AuraSrUpscaleProvider",
        model={"hub_id": "fal/AuraSR-v2", "seed": 7, "max_input_side": max_input_side},
    ))


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_weights_load_from_the_downloaded_snapshot(fakes):
    _, snapshot = fakes
    provider = _provider()

    await provider.load("/models")

    assert _FakeAuraSr.loaded_from == [str(snapshot / "model.safetensors")]
    assert provider.is_loaded()


@pytest.mark.asyncio
async def test_upscale_is_4x_png_with_the_configured_seed(fakes):
    seeds, _ = fakes
    provider = _provider()
    await provider.load("/models")

    out = Image.open(io.BytesIO(await provider.upscale(_png(30, 20))))

    assert (out.format, out.size) == ("PNG", (120, 80))
    assert seeds == [7]


@pytest.mark.asyncio
async def test_an_input_over_the_limit_is_refused(fakes):
    provider = _provider(max_input_side=64)
    await provider.load("/models")

    with pytest.raises(ValueError, match="max_input_side"):
        await provider.upscale(_png(100, 50))
