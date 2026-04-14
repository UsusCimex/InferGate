"""Dependency injection via FastAPI's Depends + app.state."""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from app.services.cache_manager import CacheManager
    from app.services.gpu_scheduler import GpuScheduler
    from app.services.provider_manager import ProviderManager


def get_provider_manager(request: Request) -> ProviderManager:
    return request.app.state.provider_manager


def get_gpu_scheduler(request: Request) -> GpuScheduler:
    return request.app.state.gpu_scheduler


def get_cache_manager(request: Request) -> CacheManager:
    return request.app.state.cache_manager


def get_defaults(request: Request) -> dict[str, str]:
    return request.app.state.defaults


def get_start_time(request: Request) -> float:
    return request.app.state.start_time
