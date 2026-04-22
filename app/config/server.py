"""Server-level configuration (global settings from server.yaml)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.config.enums import EvictionPolicy, LogLevel


class AuthConfig(BaseModel):
    enabled: bool = False
    api_keys: list[str] = []


class GpuConfig(BaseModel):
    # Legacy count-based LRU ceiling — kept as a secondary guard when the
    # byte-budget LRU can't evict (all loaded models are pinned).
    max_loaded_models: int = Field(3, ge=1)
    # Byte-budget LRU: when summed declared vram_mb of loaded models
    # would exceed this, evict LRU until it fits. 0 = disabled (count-
    # based only). Set to your GPU VRAM minus a safety margin (e.g. on a
    # 12 GB card use ~10000, keeping 2 GB for activations and spikes).
    max_vram_budget_mb: int = Field(0, ge=0)
    # Defaults assume the budget-based LRU is on and leaves 2 GB headroom
    # for activations that exceed the declared static cost. Raise to 0.90
    # on a 24 GB card where the absolute margin is bigger.
    vram_headroom_mb: int = Field(0, ge=0)
    pinned_models: list[str] = []
    device: str = "cuda:0"
    # Watchdog (Commit 5) — params here so all GPU tuning lives together.
    # Disabled by default (interval=0); set to 10-30s in production.
    watchdog_interval_seconds: int = Field(0, ge=0)
    # When VRAM usage exceeds this fraction of total (0.0-1.0), the
    # watchdog triggers emergency eviction of the LRU loaded model.
    watchdog_vram_threshold: float = Field(0.92, ge=0.5, le=1.0)
    # Host RAM watchdog — logs warning (no eviction, we don't own host
    # processes) when the host's `used / total` crosses this.
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
