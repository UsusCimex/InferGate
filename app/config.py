from __future__ import annotations

import os
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class AuthConfig(BaseModel):
    enabled: bool = False
    api_keys: list[str] = []


class GpuConfig(BaseModel):
    max_loaded_models: int = 3
    pinned_models: list[str] = []
    device: str = "cuda:0"


class QueueConfig(BaseModel):
    max_size: int = 50
    default_timeout_seconds: int = 120


class CacheGlobalConfig(BaseModel):
    enabled: bool = True
    directory: str = "./cache"
    max_total_size_gb: float = 10
    eviction_policy: str = "lru"
    cleanup_interval_minutes: int = 30


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
    requests_per_minute: int = 60


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    workers: int = 1
    auth: AuthConfig = AuthConfig()
    gpu: GpuConfig = GpuConfig()
    queue: QueueConfig = QueueConfig()
    cache: CacheGlobalConfig = CacheGlobalConfig()
    cors: CorsConfig = CorsConfig()
    models_dir: str = "./models"
    defaults: DefaultsConfig = DefaultsConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()


class ModelCacheConfig(BaseModel):
    enabled: bool = False
    strategy: str = "never"
    ttl_hours: float | None = None
    max_size_mb: int = 0


class ModelQueueConfig(BaseModel):
    priority: str = "medium"
    timeout_seconds: int = 120
    max_concurrent: int = 1


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
    model: dict[str, Any] = {}
    cache: ModelCacheConfig = ModelCacheConfig()
    queue: ModelQueueConfig = ModelQueueConfig()
    metadata: ModelMetadata = ModelMetadata()


def load_server_config(path: str | Path = "config/server.yaml") -> ServerConfig:
    path = Path(path)
    if not path.exists():
        return ServerConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ServerConfig(**data)


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
            print(f"Warning: failed to load {yaml_file}: {e}")
    return configs
