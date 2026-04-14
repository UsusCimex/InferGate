from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Enums for validated choice fields ---


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EvictionPolicy(str, Enum):
    LRU = "lru"


class CacheStrategy(str, Enum):
    ALWAYS = "always"
    SEED_ONLY = "seed_only"
    NEVER = "never"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --- Server-level configuration ---


class AuthConfig(BaseModel):
    enabled: bool = False
    api_keys: list[str] = []


class GpuConfig(BaseModel):
    max_loaded_models: int = Field(3, ge=1)
    pinned_models: list[str] = []
    device: str = "cuda:0"


class QueueConfig(BaseModel):
    max_size: int = Field(50, ge=1)


class CacheConfig(BaseModel):
    enabled: bool = True
    directory: str = "./cache"
    max_total_size_gb: float = Field(10, ge=0)
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    cleanup_interval_minutes: int = Field(30, ge=1)


class CorsConfig(BaseModel):
    allow_origins: list[str] = ["*"]
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]


class DefaultsConfig(BaseModel):
    image: str = "flux1-schnell"
    text: str = "qwen3.5-9b"
    tts: str = "kokoro-82m"


class RateLimitConfig(BaseModel):
    enabled: bool = False
    requests_per_minute: int = Field(60, ge=1)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    log_level: LogLevel = LogLevel.INFO
    workers: int = Field(1, ge=1)
    auth: AuthConfig = AuthConfig()
    gpu: GpuConfig = GpuConfig()
    queue: QueueConfig = QueueConfig()
    cache: CacheConfig = CacheConfig()
    cors: CorsConfig = CorsConfig()
    models_dir: str = "./models"
    defaults: DefaultsConfig = DefaultsConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()


# --- Model-level configuration ---


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


# --- Config loaders ---


def load_server_config(path: str | Path = "config/server.yaml") -> ServerConfig:
    path = Path(path)
    if not path.exists():
        return ServerConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ServerConfig(**data)


def load_single_model_config(path: str | Path) -> ModelConfig:
    """Load a single model config from a YAML file (used by workers)."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ModelConfig(**data)


def load_model_configs(models_dir: str | Path = "config/models") -> list[ModelConfig]:
    models_dir = Path(models_dir)
    configs = []
    if not models_dir.exists():
        return configs
    for yaml_file in sorted(models_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f) or {}
        try:
            config = ModelConfig(**data)
            configs.append(config)
        except Exception as e:
            logger.warning("Failed to load %s: %s", yaml_file, e)
    return configs
