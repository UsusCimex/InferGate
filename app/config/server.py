from __future__ import annotations

from pydantic import BaseModel, Field

from app.config.enums import EvictionPolicy, LogLevel


class AuthConfig(BaseModel):
    enabled: bool = False
    api_keys: list[str] = []


class GpuConfig(BaseModel):
    max_loaded_models: int = Field(3, ge=1)
    max_vram_budget_mb: int = Field(0, ge=0)
    vram_headroom_mb: int = Field(0, ge=0)
    pinned_models: list[str] = []
    category_reservations: dict[str, int] = {}
    device: str = "cuda:0"
    watchdog_interval_seconds: int = Field(0, ge=0)
    watchdog_vram_threshold: float = Field(0.92, ge=0.5, le=1.0)
    watchdog_ram_threshold: float = Field(0.90, ge=0.5, le=1.0)


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
    embedding_text: str | None = None
    embedding_audio: str | None = None
    embedding_image: str | None = None
    embedding_video: str | None = None


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
