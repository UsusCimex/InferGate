"""Tests for MemoryWatchdog: drive scan_once() directly to avoid sleep loops."""
from __future__ import annotations

import pytest

from app.services.memory_watchdog import MemoryWatchdog


class _StubProvider:
    """Minimal provider that reports preset stats, enough for the watchdog
    aggregation + eviction logic."""
    def __init__(
        self, vram_used_mb: int, vram_total_mb: int, loaded: bool = True,
        declared_mb: int | None = None, gpu: int = 0,
    ):
        self.gpu = gpu
        self._vram_used = vram_used_mb
        self._vram_total = vram_total_mb
        self._declared = vram_used_mb if declared_mb is None else declared_mb
        self._loaded = loaded
        self.unload_calls = 0

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vram_mb(self) -> int:
        return self._declared

    async def get_stats(self) -> dict:
        return {"vram_used_mb": self._vram_used, "vram_total_mb": self._vram_total}

    async def unload(self):
        self._loaded = False
        self.unload_calls += 1


class _StubManager:
    def __init__(self, loaded: dict[str, _StubProvider], pinned: set | None = None):
        self._registry = dict(loaded)
        self._loaded_order = dict.fromkeys(loaded)
        self._pinned = pinned or set()

    def loaded_models(self) -> list[str]:
        return list(self._loaded_order)

    def get(self, model_id):
        return self._registry[model_id]

    def gpu_of(self, model_id):
        return self._registry[model_id].gpu

    def _find_lru_victim(self, gpu=None):
        for m in self._loaded_order:
            if m not in self._pinned and (gpu is None or self.gpu_of(m) == gpu):
                return m
        return None

    async def unload_model(self, model_id):
        await self._registry[model_id].unload()
        self._loaded_order.pop(model_id, None)


@pytest.mark.asyncio
async def test_watchdog_evicts_on_vram_over_threshold():
    """live VRAM >= threshold * total: evict LRU non-pinned model."""
    providers = {
        "old": _StubProvider(vram_used_mb=11000, vram_total_mb=12000),  # 91%
        "new": _StubProvider(vram_used_mb=11000, vram_total_mb=12000),  # same total
    }
    manager = _StubManager(providers)
    wd = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.9, ram_threshold=0.99)

    summary = await wd.scan_once()
    assert summary["vram_over_threshold"] is True
    assert summary["evicted"] == ["old"]  # LRU first
    assert providers["old"].unload_calls == 1


@pytest.mark.asyncio
async def test_watchdog_noop_below_threshold():
    """Usage under threshold: no eviction, no logs."""
    providers = {
        "m1": _StubProvider(vram_used_mb=5000, vram_total_mb=12000),  # 42%
    }
    manager = _StubManager(providers)
    wd = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.9, ram_threshold=0.99)

    summary = await wd.scan_once()
    assert summary["vram_over_threshold"] is False
    assert summary["evicted"] == []
    assert providers["m1"].unload_calls == 0


@pytest.mark.asyncio
async def test_watchdog_cannot_evict_when_all_pinned(caplog):
    """Over threshold + all pinned: warning, no eviction."""
    providers = {
        "pinned-a": _StubProvider(vram_used_mb=11500, vram_total_mb=12000),
        "pinned-b": _StubProvider(vram_used_mb=11500, vram_total_mb=12000),
    }
    manager = _StubManager(providers, pinned={"pinned-a", "pinned-b"})
    wd = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.9, ram_threshold=0.99)

    with caplog.at_level("WARNING"):
        summary = await wd.scan_once()
    assert summary["vram_over_threshold"] is True
    assert summary["evicted"] == []
    assert all(p.unload_calls == 0 for p in providers.values())
    assert any("all loaded models are pinned" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_watchdog_keeps_lone_model_that_reserves_the_gpu(caplog):
    """vLLM reserves ~90% of VRAM up front: a single loaded model over threshold stays loaded."""
    providers = {"vllm": _StubProvider(vram_used_mb=11850, vram_total_mb=12227)}
    manager = _StubManager(providers)
    wd = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.92, ram_threshold=0.99)

    with caplog.at_level("WARNING"):
        summary = await wd.scan_once()
    assert summary["vram_over_threshold"] is True
    assert summary["evicted"] == []
    assert providers["vllm"].unload_calls == 0
    assert not any("VRAM" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_watchdog_keeps_cpu_models_next_to_a_lone_gpu_model():
    """A CPU model beside vLLM frees no VRAM, so the sweep evicts nothing."""
    providers = {
        "clip-on-cpu": _StubProvider(vram_used_mb=11700, vram_total_mb=12227, declared_mb=0),
        "vllm": _StubProvider(vram_used_mb=11700, vram_total_mb=12227),
    }
    manager = _StubManager(providers)
    wd = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.92, ram_threshold=0.99)

    summary = await wd.scan_once()
    assert summary["vram_over_threshold"] is True
    assert summary["evicted"] == []
    assert all(p.unload_calls == 0 for p in providers.values())


@pytest.mark.asyncio
async def test_watchdog_ram_threshold_is_advisory(caplog, monkeypatch):
    """Host RAM overshoot: log warning but no eviction (we don't own
    host processes)."""
    import psutil

    class _FakeVM:
        total = 16 * 1024 * 1024 * 1024       # 16 GB
        available = 1 * 1024 * 1024 * 1024    # 1 GB free, 94% used
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVM())

    providers = {"m1": _StubProvider(vram_used_mb=100, vram_total_mb=12000)}
    manager = _StubManager(providers)
    wd = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.99, ram_threshold=0.9)

    with caplog.at_level("WARNING"):
        summary = await wd.scan_once()
    assert summary["ram_over_threshold"] is True
    assert summary["evicted"] == []
    assert providers["m1"].unload_calls == 0
    assert any("host RAM" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_watchdog_handles_empty_stats():
    """Worker returning {} (transport failure) shouldn't crash the sweep."""
    class _EmptyProvider(_StubProvider):
        async def get_stats(self):
            return {}

    providers = {"dead": _EmptyProvider(0, 0)}
    manager = _StubManager(providers)
    wd = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.9, ram_threshold=0.99)

    summary = await wd.scan_once()
    # No data: no eviction, no thresholds tripped
    assert summary["vram_over_threshold"] is False
    assert summary["evicted"] == []


@pytest.mark.asyncio
async def test_watchdog_lifecycle():
    """start() is idempotent; stop() cancels; interval=0 skips start."""
    import asyncio

    providers = {"m": _StubProvider(0, 12000)}
    manager = _StubManager(providers)

    # Disabled: no task
    wd_disabled = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.9, ram_threshold=0.99)
    wd_disabled.start()
    assert wd_disabled._task is None

    # Enabled: task starts, double-start is no-op, stop cancels
    wd = MemoryWatchdog(manager, interval_seconds=60, vram_threshold=0.9, ram_threshold=0.99)
    wd.start()
    first = wd._task
    wd.start()
    assert wd._task is first
    await asyncio.sleep(0)  # let scheduler see it
    await wd.stop()
    assert wd._task is None
    assert first.cancelled() or first.done()


@pytest.mark.asyncio
async def test_watchdog_judges_each_gpu_on_its_own():
    """A lone vLLM model filling GPU 0 must not cost a model on GPU 1 its slot."""
    providers = {
        "llm": _StubProvider(vram_used_mb=11500, vram_total_mb=12000, gpu=0),
        "image-a": _StubProvider(vram_used_mb=6000, vram_total_mb=24000, gpu=1),
        "image-b": _StubProvider(vram_used_mb=6000, vram_total_mb=24000, gpu=1),
    }
    manager = _StubManager(providers)
    wd = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.9, ram_threshold=0.99)

    summary = await wd.scan_once()
    assert summary["vram_over_threshold"] is True
    assert summary["evicted"] == []
    assert summary["vram_used_mb"] == 11500 + 6000
    assert summary["vram_total_mb"] == 12000 + 24000


@pytest.mark.asyncio
async def test_watchdog_evicts_on_the_gpu_over_threshold():
    providers = {
        "llm": _StubProvider(vram_used_mb=3000, vram_total_mb=12000, gpu=0),
        "image-a": _StubProvider(vram_used_mb=23000, vram_total_mb=24000, gpu=1),
        "image-b": _StubProvider(vram_used_mb=23000, vram_total_mb=24000, gpu=1),
    }
    manager = _StubManager(providers)
    wd = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.9, ram_threshold=0.99)

    summary = await wd.scan_once()
    assert summary["evicted"] == ["image-a"]
    assert providers["llm"].unload_calls == 0


@pytest.mark.asyncio
async def test_watchdog_reports_every_gpu_it_evicted_from():
    providers = {
        "text-a": _StubProvider(vram_used_mb=11000, vram_total_mb=12000, gpu=0),
        "image-a": _StubProvider(vram_used_mb=23000, vram_total_mb=24000, gpu=1),
        "text-b": _StubProvider(vram_used_mb=11000, vram_total_mb=12000, gpu=0),
        "image-b": _StubProvider(vram_used_mb=23000, vram_total_mb=24000, gpu=1),
    }
    manager = _StubManager(providers)
    wd = MemoryWatchdog(manager, interval_seconds=0, vram_threshold=0.9, ram_threshold=0.99)

    summary = await wd.scan_once()
    assert summary["evicted"] == ["text-a", "image-a"]
