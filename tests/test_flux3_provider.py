"""Flux3ImageProvider against a fake diffusers: no prompt upsampler, quantized parts, the paged text encoder, the call."""
from __future__ import annotations

import base64
import io
import sys
import types
from typing import Any, ClassVar

import pytest
from PIL import Image

from app.config import ModelConfig
from app.providers.image import flux3_provider
from app.providers.image.flux3_provider import Flux3ImageProvider


class _Generator:
    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


class _Module:
    def __init__(self, name: str) -> None:
        self.name = name
        self.device = "cpu"

    def to(self, device: str) -> _Module:
        self.device = device
        return self


class _TextPipeline:
    created: ClassVar[list[_TextPipeline]] = []

    def __init__(self) -> None:
        self.loaded: list[str] | None = None
        self.text_encoder: _Module | None = None
        self.calls: list[tuple[str, str]] = []
        _TextPipeline.created.append(self)

    def load_components(self, names: list[str] | None = None, **kwargs: Any) -> None:
        self.loaded = names
        if names is None:
            self.text_encoder = _Module("bf16-text-encoder")

    def update_components(self, text_encoder: Any) -> None:
        self.text_encoder = text_encoder

    def __call__(self, prompt: str, output: str) -> str:
        self.calls.append((prompt, self.text_encoder.device))
        return f"embeds of {prompt}"


class _TextBlock:
    def init_pipeline(self, hub_id: str) -> _TextPipeline:
        return _TextPipeline()


class _Blocks:
    def __init__(self) -> None:
        self.sub_blocks = {"prompt_upsampler": object(), "text_encoder": _TextBlock(), "denoise": object()}

    def init_pipeline(self, hub_id: str) -> _Pipeline:
        return _Pipeline(self)


class _Pipeline:
    created: ClassVar[list[_Pipeline]] = []

    def __init__(self, blocks: _Blocks) -> None:
        self.blocks = blocks
        self.loaded: list[str] | None = None
        self.updated: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self.device = "cpu"
        _Pipeline.created.append(self)

    def load_components(self, names: list[str] | None = None, **kwargs: Any) -> None:
        self.loaded = names

    def update_components(self, **components: Any) -> None:
        self.updated.update(components)

    def to(self, device: str) -> None:
        self.device = device

    def __call__(self, **kwargs: Any) -> list[Image.Image]:
        self.calls.append(kwargs)
        return [Image.new("RGB", (kwargs.get("width", 64), kwargs.get("height", 64)), (10, 20, 30))]


class _Transformer:
    files: ClassVar[list[str]] = []

    @classmethod
    def from_single_file(cls, path: str, **kwargs: Any) -> str:
        cls.files.append(path)
        return "fp8r-transformer"


class _Pager:
    def __init__(self, module: _Module) -> None:
        self.module = module.to("cpu")

    def page_in(self) -> None:
        self.module.to("cuda")

    def page_out(self) -> None:
        self.module.to("cpu")


class _Qwen3VL:
    configs: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, hub_id: str, quantization_config: dict[str, Any], **kwargs: Any) -> _Module:
        cls.configs.append(quantization_config)
        return _Module("nf4-text-encoder").to("cuda")


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    _Pipeline.created = []
    _TextPipeline.created = []
    _Transformer.files = []
    _Qwen3VL.configs = []
    diffusers = types.SimpleNamespace(
        ModularPipeline=types.SimpleNamespace(from_pretrained=lambda *a, **kw: types.SimpleNamespace(blocks=_Blocks())),
        Flux3Transformer2DModel=_Transformer,
        BFLQuantizationConfig=lambda **kw: kw,
    )
    transformers = types.SimpleNamespace(BitsAndBytesConfig=lambda **kw: kw, Qwen3VLForConditionalGeneration=_Qwen3VL)
    torch = types.SimpleNamespace(
        float32="float32", bfloat16="bfloat16", Generator=_Generator,
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(flux3_provider, "_PinnedPager", _Pager)


def _provider(**model: Any) -> Flux3ImageProvider:
    return Flux3ImageProvider(ModelConfig(
        id="flux3-klein", display_name="FLUX 3 Image Klein", category="image", provider_class="Flux3ImageProvider",
        model={"hub_id": "black-forest-labs/flux-3-image-klein", "revision": "main", "warmup": False,
               "default_params": {"num_inference_steps": 16, "width": 1024, "height": 1024}, **model},
    ))


def _b64_png() -> str:
    buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (200, 30, 30, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def test_the_fp8r_transformer_replaces_the_bf16_one_and_the_upsampler_is_left_out():
    await _provider(transformer_variant="fp8r").load("/models")

    pipe = _Pipeline.created[0]
    assert set(pipe.blocks.sub_blocks) == {"denoise"}
    assert pipe.loaded == ["vae"]
    assert pipe.updated == {"transformer": "fp8r-transformer"}
    assert _Transformer.files == [
        "https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/variants/fp8r/model.safetensors"
    ]
    assert pipe.device == "cuda"
    assert _TextPipeline.created[0].text_encoder.device == "cuda"


async def test_without_quantization_every_part_loads_from_the_repository():
    await _provider().load("/models")

    assert _Pipeline.created[0].loaded is None
    assert _Pipeline.created[0].updated == {}
    assert _TextPipeline.created[0].loaded is None
    assert _Qwen3VL.configs == []


async def test_an_unknown_variant_fails_the_load():
    with pytest.raises(ValueError, match="transformer_variant"):
        await _provider(transformer_variant="int3").load("/models")


async def test_a_paged_nf4_text_encoder_is_on_the_gpu_only_while_it_encodes():
    provider = _provider(text_encoder_quantization="nf4", page_text_encoders=True)
    await provider.load("/models")
    text_pipe = _TextPipeline.created[0]

    assert text_pipe.loaded == ["tokenizer"]
    assert _Qwen3VL.configs[0]["bnb_4bit_quant_type"] == "nf4"
    assert text_pipe.text_encoder.device == "cpu"

    await provider.generate("A red fox")

    assert text_pipe.calls == [("A red fox", "cuda")]
    assert text_pipe.text_encoder.device == "cpu"


async def test_a_reference_larger_than_the_cap_is_scaled_down_with_its_aspect():
    provider = _provider(reference_max_side=16)
    await provider.load("/models")

    await provider.generate("A red fox", reference_images=[_b64_png()])

    assert [image.size for image in _Pipeline.created[0].calls[0]["image"]] == [(16, 16)]


async def test_the_caption_is_encoded_without_weights_and_the_rest_goes_to_the_denoiser():
    provider = _provider()
    await provider.load("/models")

    png = await provider.generate("(A red fox:1.4) in snow", size="512x768", seed=7, reference_images=[_b64_png()],
                                  response_format="b64_json")

    assert _TextPipeline.created[0].calls[0][0] == "A red fox in snow"
    call = _Pipeline.created[0].calls[0]
    assert call["prompt_embeds"] == "embeds of A red fox in snow"
    assert "prompt" not in call
    assert (call["width"], call["height"], call["num_inference_steps"]) == (512, 768, 16)
    assert call["generator"].seed == 7
    assert [image.mode for image in call["image"]] == ["RGB"]
    assert call["output"] == "images"
    assert "response_format" not in call
    assert Image.open(io.BytesIO(png)).size == (512, 768)


async def test_an_unknown_vae_dtype_fails_the_load():
    with pytest.raises(ValueError, match="vae_dtype"):
        await _provider(vae_dtype="float8").load("/models")


async def test_an_empty_scheduler_or_lora_list_means_the_model_default():
    provider = _provider()
    await provider.load("/models")

    await provider.generate("A red fox", scheduler="", loras=[], negative_prompt=None)

    call = _Pipeline.created[0].calls[0]
    assert not {"scheduler", "loras", "negative_prompt"} & set(call)


@pytest.mark.parametrize("param", ["guidance_scale", "negative_prompt", "image", "loras"])
async def test_what_the_distilled_model_cannot_do_is_a_bad_request(param):
    provider = _provider()
    await provider.load("/models")

    with pytest.raises(ValueError, match=param):
        await provider.generate("A red fox", **{param: "x"})
