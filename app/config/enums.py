"""Validated enum types used across server and model configuration."""
from __future__ import annotations

from enum import StrEnum


class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EvictionPolicy(StrEnum):
    LRU = "lru"


class CacheStrategy(StrEnum):
    ALWAYS = "always"
    SEED_ONLY = "seed_only"
    NEVER = "never"


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
