from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config.enums import CacheStrategy, Priority

# model_id is a path component under the cache directory — keep it traversal-safe.
_MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class ModelCacheConfig(BaseModel):
    """Per-model cache policy."""
    enabled: bool = False
    strategy: CacheStrategy = CacheStrategy.NEVER
    ttl_hours: float | None = None
    max_size_mb: int = Field(0, ge=0)


class ModelQueueConfig(BaseModel):
    """Per-model scheduler queue policy."""
    priority: Priority = Priority.MEDIUM
    timeout_seconds: int = Field(120, ge=1)
    max_concurrent: int = Field(1, ge=1)


class ModelMetadata(BaseModel):
    """Human-facing model metadata surfaced in /v1/models."""
    license: str = ""
    description: str = ""
    tags: list[str] = []


class ModelCapabilities(BaseModel):
    """Per-model feature flags enforced at the API boundary."""
    voice_clone_only: bool = False


class ModelConfig(BaseModel):
    """A single model's full configuration as loaded from YAML."""
    id: str = Field(pattern=_MODEL_ID_PATTERN, max_length=128)
    display_name: str
    category: str
    provider_class: str
    enabled: bool = True
    worker_url: str | None = None
    model: dict[str, Any] = {}
    cache: ModelCacheConfig = ModelCacheConfig()
    queue: ModelQueueConfig = ModelQueueConfig()
    metadata: ModelMetadata = ModelMetadata()
    capabilities: ModelCapabilities = ModelCapabilities()
