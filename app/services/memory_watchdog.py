from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.provider_manager import ProviderManager

logger = logging.getLogger(__name__)


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
            "evicted": None,
        }

        loaded = list(self._manager.loaded_models())
        agg_used = 0
        agg_total = 0
        worker_stats: dict[str, dict] = {}
        for model_id in loaded:
            try:
                provider = self._manager.get(model_id)
            except Exception:
                continue
            try:
                stats = await provider.get_stats()
            except Exception as e:
                logger.debug("get_stats(%s) failed: %s", model_id, e)
                continue
            if not stats:
                continue
            worker_stats[model_id] = stats
            # Workers on one host share a GPU — max of reported totals is the physical total.
            agg_used = max(agg_used, stats.get("vram_used_mb", 0))
            agg_total = max(agg_total, stats.get("vram_total_mb", 0))

        summary["vram_used_mb"] = agg_used
        summary["vram_total_mb"] = agg_total

        ram_used, ram_total = self._host_ram_snapshot()
        summary["ram_used_mb"] = ram_used
        summary["ram_total_mb"] = ram_total

        if agg_total > 0 and agg_used >= agg_total * self._vram_threshold:
            summary["vram_over_threshold"] = True
            logger.warning(
                "MemoryWatchdog: VRAM %d/%d MB (%.0f%%) — over threshold %.0f%%; evicting LRU",
                agg_used, agg_total, 100 * agg_used / agg_total,
                100 * self._vram_threshold,
            )
            victim = self._manager._find_lru_victim()
            if victim is not None:
                try:
                    await self._manager.unload_model(victim)
                    summary["evicted"] = victim
                    logger.info("MemoryWatchdog: evicted %s under VRAM pressure", victim)
                except Exception as e:
                    logger.error("MemoryWatchdog: eviction of %s failed: %s", victim, e)
            else:
                logger.warning(
                    "MemoryWatchdog: VRAM over threshold but all loaded models are pinned or busy"
                )

        # Host RAM is advisory — the watchdog owns no host processes.
        if ram_total > 0 and ram_used >= ram_total * self._ram_threshold:
            summary["ram_over_threshold"] = True
            logger.warning(
                "MemoryWatchdog: host RAM %d/%d MB (%.0f%%) — over threshold %.0f%%; "
                "host-level OOM risk",
                ram_used, ram_total, 100 * ram_used / ram_total,
                100 * self._ram_threshold,
            )

        return summary

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
