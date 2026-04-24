"""Model-level configuration (per-model YAML files)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config.enums import CacheStrategy, Priority

# model_id is used as a path component in the cache directory, so it must be
# a strict identifier with no separators or traversal sequences.
_MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


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


class ModelCapabilities(BaseModel):
    # TTS providers that refuse to synthesise without a `reference_audio`
    # clip (XTTS-v2, Qwen3-TTS). The gateway fails fast with a 400 on the
    # flat /v1/audio/speech endpoint and steers callers to /voice-clone
    # instead of paying a round-trip to the worker just to get a 400 back.
    voice_clone_only: bool = False


class ModelConfig(BaseModel):
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
