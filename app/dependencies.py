"""Dependency injection via FastAPI's Depends + app.state."""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Header, Request

from app.config import Priority

if TYPE_CHECKING:
    from app.config import UploadLimitsConfig
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


def get_upload_limits(request: Request) -> UploadLimitsConfig:
    return request.app.state.upload_limits


def get_allowed_adapter_repos(request: Request) -> list[str]:
    return request.app.state.allowed_adapter_repos


def get_priority(
    priority: Priority | None = Header(None, alias="X-InferGate-Priority"),
) -> Priority | None:
    """Queue priority of this request; None keeps the model's `queue.priority`."""
    return priority
