from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.provider_manager import ProviderManager

logger = logging.getLogger(__name__)


async def live_vram_by_gpu(manager: ProviderManager) -> dict[int, tuple[int, int]]:
    """Used and total MB of each GPU from the loaded models' `/stats`; workers on one GPU share its numbers."""
    by_gpu: dict[int, tuple[int, int]] = {}
    for model_id in list(manager.loaded_models()):
        try:
            stats = await manager.get(model_id).get_stats()
            gpu = manager.gpu_of(model_id)
        except Exception as e:
            logger.debug("get_stats(%s) failed: %s", model_id, e)
            continue
        if stats:
            used, total = by_gpu.get(gpu, (0, 0))
            by_gpu[gpu] = (
                max(used, stats.get("vram_used_mb", 0)),
                max(total, stats.get("vram_total_mb", 0)),
            )
    return by_gpu


async def live_vram(manager: ProviderManager) -> tuple[int, int]:
    """Used and total MB summed over the GPUs of the loaded models."""
    by_gpu = await live_vram_by_gpu(manager)
    return sum(used for used, _ in by_gpu.values()), sum(total for _, total in by_gpu.values())


class MemoryWatchdog:
    """Background probe that evicts LRU models when live VRAM exceeds a threshold."""

    def __init__(
        self,
        manager: ProviderManager,
        interval_seconds: int,
        vram_threshold: float,
        ram_threshold: float,
    ) -> None:
        self._manager = manager
        self._interval = interval_seconds
        self._vram_threshold = vram_threshold
        self._ram_threshold = ram_threshold
        self._task: asyncio.Task | None = None
        self._last_evict_tick = 0

    def start(self) -> None:
        if self._interval <= 0:
            logger.info("MemoryWatchdog disabled (interval_seconds=0)")
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())
        logger.info(
            "MemoryWatchdog started: interval=%ds, vram>%d%%, ram>%d%%",
            self._interval,
            int(self._vram_threshold * 100),
            int(self._ram_threshold * 100),
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        try:
            while True:
                try:
                    await self.scan_once()
                except Exception as e:
                    logger.exception("MemoryWatchdog scan failed: %s", e)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            logger.info("MemoryWatchdog stopped")
            raise

    async def scan_once(self) -> dict:
        """Run one sweep; return a summary so tests can drive cycles deterministically."""
        summary: dict = {
            "vram_used_mb": 0,
            "vram_total_mb": 0,
            "ram_used_mb": 0,
            "ram_total_mb": 0,
            "vram_over_threshold": False,
            "ram_over_threshold": False,
            "evicted": [],
        }

        gpu_models: dict[int, int] = {}
        for model_id in list(self._manager.loaded_models()):
            with contextlib.suppress(Exception):
                if self._manager.get(model_id).vram_mb > 0:
                    gpu = self._manager.gpu_of(model_id)
                    gpu_models[gpu] = gpu_models.get(gpu, 0) + 1
        by_gpu = await live_vram_by_gpu(self._manager)
        summary["vram_used_mb"] = sum(used for used, _ in by_gpu.values())
        summary["vram_total_mb"] = sum(total for _, total in by_gpu.values())

        ram_used, ram_total = self._host_ram_snapshot()
        summary["ram_used_mb"] = ram_used
        summary["ram_total_mb"] = ram_total

        for gpu, (used, total) in sorted(by_gpu.items()):
            if total <= 0 or used < total * self._vram_threshold:
                continue
            summary["vram_over_threshold"] = True
            # A lone GPU model can't be crowding out another one, and engines like vLLM reserve most
            # of the GPU up front: evicting it would only force a reload on the next request. Models
            # on the CPU hold no VRAM to free.
            if gpu_models.get(gpu, 0) > 1:
                await self._evict_from(gpu, used, total, summary)

        # Host RAM is advisory: the watchdog owns no host processes.
        if ram_total > 0 and ram_used >= ram_total * self._ram_threshold:
            summary["ram_over_threshold"] = True
            logger.warning(
                "MemoryWatchdog: host RAM %d/%d MB (%.0f%%) over threshold %.0f%%; "
                "host-level OOM risk",
                ram_used, ram_total, 100 * ram_used / ram_total,
                100 * self._ram_threshold,
            )

        return summary

    async def _evict_from(self, gpu: int, used: int, total: int, summary: dict) -> None:
        logger.warning(
            "MemoryWatchdog: GPU %d VRAM %d/%d MB (%.0f%%) over threshold %.0f%%; evicting LRU",
            gpu, used, total, 100 * used / total, 100 * self._vram_threshold,
        )
        victim = self._manager._find_lru_victim(gpu=gpu)
        if victim is None:
            logger.warning(
                "MemoryWatchdog: VRAM over threshold but all loaded models are pinned or busy (GPU %d)", gpu
            )
            return
        try:
            await self._manager.unload_model(victim)
            summary["evicted"].append(victim)
            logger.info("MemoryWatchdog: evicted %s under VRAM pressure", victim)
        except Exception as e:
            logger.error("MemoryWatchdog: eviction of %s failed: %s", victim, e)

    @staticmethod
    def _host_ram_snapshot() -> tuple[int, int]:
        """Return (used_mb, total_mb) for host RAM; (0, 0) when psutil is absent."""
        try:
            import psutil

            vm = psutil.virtual_memory()
            return (
                (vm.total - vm.available) // (1024 * 1024),
                vm.total // (1024 * 1024),
            )
        except ImportError:
            return 0, 0
        except Exception:
            return 0, 0
