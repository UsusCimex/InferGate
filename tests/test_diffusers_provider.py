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
