"""Global dependency container. Initialized during app lifespan."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.cache_manager import CacheManager
    from app.services.gpu_scheduler import GpuScheduler
    from app.services.provider_manager import ProviderManager

_provider_manager: ProviderManager | None = None
_gpu_scheduler: GpuScheduler | None = None
_cache_manager: CacheManager | None = None
_defaults: dict[str, str] = {}
_start_time: float = 0.0


def init_dependencies(
    provider_manager: ProviderManager,
    gpu_scheduler: GpuScheduler,
    cache_manager: CacheManager,
    defaults: dict[str, str],
    start_time: float,
) -> None:
    global _provider_manager, _gpu_scheduler, _cache_manager, _defaults, _start_time
    _provider_manager = provider_manager
    _gpu_scheduler = gpu_scheduler
    _cache_manager = cache_manager
    _defaults = defaults
    _start_time = start_time


def get_provider_manager() -> ProviderManager:
    assert _provider_manager is not None, "ProviderManager not initialized"
    return _provider_manager


def get_gpu_scheduler() -> GpuScheduler:
    assert _gpu_scheduler is not None, "GpuScheduler not initialized"
    return _gpu_scheduler


def get_cache_manager() -> CacheManager:
    assert _cache_manager is not None, "CacheManager not initialized"
    return _cache_manager


def get_defaults() -> dict[str, str]:
    return _defaults


def get_start_time() -> float:
    return _start_time
