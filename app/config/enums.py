"""Validated enum types used across server and model configuration."""
from __future__ import annotations

from enum import Enum


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
