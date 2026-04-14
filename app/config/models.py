"""Model-level configuration (per-model YAML files)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config.enums import CacheStrategy, Priority


class ModelCacheConfig(BaseModel):
    enabled: bool = False
    strategy: CacheStrategy = CacheStrategy.NEVER
    ttl_hours: float | None = None
    max_size_mb: int = Field(0, ge=0)


class ModelQueueConfig(BaseModel):
    priority: Priority = Priority.MEDIUM
    timeout_seconds: int = Field(120, ge=1)
    max_concurrent: int = Field(1, ge=1)


class ModelMetadata(BaseModel):
    license: str = ""
    description: str = ""
    tags: list[str] = []


class ModelConfig(BaseModel):
    id: str
    display_name: str
    category: str
    provider_class: str
    enabled: bool = True
    worker_url: str | None = None
    model: dict[str, Any] = {}
    cache: ModelCacheConfig = ModelCacheConfig()
    queue: ModelQueueConfig = ModelQueueConfig()
    metadata: ModelMetadata = ModelMetadata()
