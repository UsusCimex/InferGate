from __future__ import annotations

import logging

import pytest

from app.config import ModelConfig
from app.services.provider_manager import ConfigError, InsufficientResourcesError, ProviderManager
from tests.conftest import FakeImageProvider


def _config(model_id: str, gpu: object = 0, vram_mb: int = 1000) -> ModelConfig:
    return ModelConfig(
        id=model_id, display_name=model_id, category="image", provider_class="FakeImageProvider",
        model={"hub_id": "test/test", "vram_mb": vram_mb, "gpu": gpu},
    )


def _manager(placement: dict[str, int], **kwargs) -> ProviderManager:
    """Manager with one 1000 MB fake model per entry of `placement` (model id to GPU)."""
    kwargs.setdefault("max_loaded", 10)
    manager = ProviderManager(model_dir=".", **kwargs)
    for model_id, gpu in placement.items():
        manager._registry[model_id] = FakeImageProvider(_config(model_id, gpu))
    return manager


async def test_a_full_gpu_evicts_only_its_own_models():
    manager = _manager({"a0": 0, "b1": 1, "c1": 1}, max_vram_budget_mb=1500)
    await manager.ensure_loaded("a0")
    await manager.ensure_loaded("b1")

    await manager.ensure_loaded("c1")
    assert manager.loaded_models() == ["a0", "c1"]


async def test_the_model_count_limit_spans_every_gpu():
    manager = _manager({"a0": 0, "b1": 1, "c1": 1}, max_loaded=2, max_vram_budget_mb=10000)
    await manager.ensure_loaded("a0")
    await manager.ensure_loaded("b1")

    await manager.ensure_loaded("c1")
    assert manager.loaded_models() == ["b1", "c1"]


async def test_a_gpu_can_have_its_own_budget():
    manager = _manager({"a1": 1, "b1": 1, "c0": 0, "d0": 0},
                       max_vram_budget_mb=1500, vram_budgets_mb={1: 3000})
    for model_id in ("a1", "b1", "c0"):
        await manager.ensure_loaded(model_id)
    assert manager.loaded_models() == ["a1", "b1", "c0"]

    await manager.ensure_loaded("d0")
    assert manager.loaded_models() == ["a1", "b1", "d0"]


async def test_a_pinned_gpu_refuses_with_its_own_numbers():
    manager = _manager({"a0": 0, "b1": 1, "c1": 1},
                       pinned=["b1"], max_vram_budget_mb=1500)
    await manager.ensure_loaded("a0")
    await manager.ensure_loaded("b1")
    with pytest.raises(InsufficientResourcesError, match="on GPU 1: 1000 MB loaded"):
        await manager.ensure_loaded("c1")
    assert manager.loaded_models() == ["a0", "b1"]


async def test_status_reports_each_gpu():
    manager = _manager({"a0": 0, "b1": 1}, max_vram_budget_mb=1500, vram_budgets_mb={2: 8000})
    await manager.ensure_loaded("a0")
    await manager.ensure_loaded("b1")

    status = manager.status_snapshot()
    assert [m["gpu"] for m in status["loaded"]] == [0, 1]
    assert status["total_declared_vram_mb"] == 2000
    assert status["gpus"] == {
        0: {"declared_vram_mb": 1000, "effective_budget_mb": 1500},
        1: {"declared_vram_mb": 1000, "effective_budget_mb": 1500},
        2: {"declared_vram_mb": 0, "effective_budget_mb": 8000},
    }
    assert manager.preview_load("a0")["gpu"] == 0


def test_validate_config_checks_each_gpu_budget():
    manager = _manager({"a1": 1, "b1": 1}, pinned=["a1", "b1"],
                       max_vram_budget_mb=10000, vram_budgets_mb={1: 1500})
    with pytest.raises(ConfigError, match=r"GPU 1 declare 2000 MB.*vram_budgets_mb\[1\]=1500"):
        manager.validate_config()


@pytest.mark.parametrize("bad", ["cuda:1", -1, 1.5, True])
def test_a_model_with_a_bad_gpu_index_is_not_registered(bad, caplog):
    manager = ProviderManager(model_dir=".", max_loaded=2)
    with caplog.at_level(logging.WARNING, logger="app.services.provider_manager"):
        manager.discover_models([_config("bad-gpu", bad)])
    assert "bad-gpu" not in manager._registry
    assert any("model.gpu must be a GPU index" in r.message for r in caplog.records)


async def test_a_reload_with_a_bad_gpu_index_keeps_the_model():
    manager = _manager({"a0": 0})
    before = manager.get("a0")
    assert await manager.reload_model(_config("a0", "cuda:1")) is False
    assert manager.get("a0") is before


def test_a_gpu_index_may_come_as_a_string():
    manager = ProviderManager(model_dir=".", max_loaded=2)
    manager.discover_models([_config("str-gpu", "1")])
    assert manager.gpu_of("str-gpu") == 1
