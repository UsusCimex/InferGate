import io
import sys
import types

import pytest
from PIL import Image

from app.config import ModelCacheConfig, ModelConfig, ModelMetadata, ModelQueueConfig
from app.providers.image._schedulers import resolve_scheduler
from app.providers.image.diffusers_provider import DiffusersImageProvider


class EulerDiscreteScheduler:
    def __init__(self, config):
        self.config = config


class FlowMatchEulerDiscreteScheduler(EulerDiscreteScheduler):
    pass


class DPMSolverSDEScheduler:
    def __init__(self, config, extra):
        self.config = config
        self.extra = extra

    @classmethod
    def from_config(cls, config, **extra):
        return cls(config, extra)


class _Pipeline:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, **kwargs):
        self.calls.append((type(self.scheduler).__name__, kwargs))
        return types.SimpleNamespace(images=[Image.new("RGB", (8, 8))])


@pytest.fixture
def fake_diffusers(monkeypatch):
    module = types.ModuleType("diffusers")
    module.DPMSolverSDEScheduler = DPMSolverSDEScheduler
    module.DPMSolverMultistepScheduler = DPMSolverSDEScheduler
    monkeypatch.setitem(sys.modules, "diffusers", module)


@pytest.fixture
def provider(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    config = ModelConfig(
        id="sdxl-test", display_name="sdxl-test", category="image",
        provider_class="DiffusersImageProvider", enabled=True,
        model={"hub_id": "test/test", "vram_mb": 1000},
        cache=ModelCacheConfig(enabled=False, strategy="never", max_size_mb=100),
        queue=ModelQueueConfig(priority="medium", timeout_seconds=30, max_concurrent=1),
        metadata=ModelMetadata(),
    )
    provider = DiffusersImageProvider(config)
    provider._pipeline = _Pipeline(EulerDiscreteScheduler({"beta_schedule": "scaled_linear"}))
    provider._default_scheduler = provider._pipeline.scheduler
    return provider


def test_no_name_keeps_the_default():
    default = EulerDiscreteScheduler({})
    assert resolve_scheduler(default, None) is default
    assert resolve_scheduler(default, "") is default


def test_named_scheduler_is_built_from_the_default_config(fake_diffusers):
    default = EulerDiscreteScheduler({"beta_schedule": "scaled_linear"})
    swapped = resolve_scheduler(default, "dpm++_2m_karras")
    assert isinstance(swapped, DPMSolverSDEScheduler)
    assert swapped.config is default.config
    assert swapped.extra == {"use_karras_sigmas": True}


def test_flow_matching_models_reject_a_swap(fake_diffusers):
    with pytest.raises(ValueError, match="flow-matching"):
        resolve_scheduler(FlowMatchEulerDiscreteScheduler({}), "dpm++_sde")


def test_unknown_scheduler_is_rejected():
    with pytest.raises(ValueError, match="Unknown scheduler"):
        resolve_scheduler(EulerDiscreteScheduler({}), "nope")


async def test_request_without_scheduler_runs_on_the_default(fake_diffusers, provider):
    await provider.generate("a cat", scheduler="dpm++_sde")
    await provider.generate("a cat")
    await provider.generate("a cat", scheduler="")
    assert [name for name, _ in provider._pipeline.calls] == [
        "DPMSolverSDEScheduler", "EulerDiscreteScheduler", "EulerDiscreteScheduler",
    ]


async def test_weights_become_plain_text_without_compel(provider):
    await provider.generate("(A single bat:1.4), an animal", negative_prompt="(blurry:1.2), text")
    _, kwargs = provider._pipeline.calls[-1]
    assert kwargs["prompt"] == "A single bat, an animal"
    assert kwargs["negative_prompt"] == "blurry, text"


class _Latents:
    """Latents of one step: every tensor operation the preview decode makes keeps the step."""

    ndim = 4

    def __init__(self, step):
        self.step = step

    def __getitem__(self, _index):
        return self

    def to(self, _dtype):
        return self

    def __truediv__(self, _factor):
        return self

    def __add__(self, _shift):
        return self


class _PreviewPipeline(_Pipeline):
    """Runs four steps through callback_on_step_end; the VAE paints step x 60 into the red channel."""

    num_timesteps = 4

    def __init__(self, scheduler):
        super().__init__(scheduler)
        self.vae = types.SimpleNamespace(
            config=types.SimpleNamespace(scaling_factor=0.13, shift_factor=None),
            dtype=None,
            decode=lambda sample, return_dict=False: (sample,),
        )
        self.image_processor = types.SimpleNamespace(
            postprocess=lambda decoded, output_type: [
                Image.new("RGB", (8, 8), (decoded.step * 60, 0, 0))
            ],
        )

    def __call__(self, callback_on_step_end=None, callback_on_step_end_tensor_inputs=None, **kwargs):
        self.calls.append((type(self.scheduler).__name__, {
            **kwargs, "callback_on_step_end_tensor_inputs": callback_on_step_end_tensor_inputs,
        }))
        for step in range(self.num_timesteps):
            if callback_on_step_end is not None:
                callback_on_step_end(self, step, 0, {"latents": _Latents(step)})
        return types.SimpleNamespace(images=[Image.new("RGB", (8, 8))])


async def test_stream_sends_vae_previews_of_evenly_spaced_steps(provider):
    provider._pipeline = _PreviewPipeline(provider._pipeline.scheduler)

    frames = [frame async for frame in provider.generate_stream("a cat", 2)]

    assert [frame.final for frame in frames] == [False, False, True]
    reds = [Image.open(io.BytesIO(frame.png)).getpixel((0, 0))[0] for frame in frames[:2]]
    assert reds == [0, 60]
    _, kwargs = provider._pipeline.calls[-1]
    assert kwargs["callback_on_step_end_tensor_inputs"] == ["latents"]


async def test_stream_without_previews_or_step_callbacks_sends_the_final_image(provider):
    frames = [frame async for frame in provider.generate_stream("a cat", 2)]

    assert [frame.final for frame in frames] == [True]
    _, kwargs = provider._pipeline.calls[-1]
    assert "callback_on_step_end" not in kwargs
