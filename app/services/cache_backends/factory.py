from __future__ import annotations

from typing import Any

from app.services.cache_backends.base import CacheBackend
from app.services.cache_backends.local import LocalCacheBackend


def make_cache_backend(global_config: dict[str, Any]) -> CacheBackend:
    """Build a CacheBackend from the cache section of server.yaml.

    Selection key: `backend` ("local" by default). Unknown values raise ValueError so
    a typo in YAML fails loud at startup instead of silently falling back.
    """
    backend = str(global_config.get("backend", "local")).lower()
    if backend == "local":
        return LocalCacheBackend(global_config)
    if backend == "redis":
        # Lazy import: redis is optional; only fail if user actually selects it.
        from app.services.cache_backends.redis import RedisCacheBackend
        return RedisCacheBackend(global_config)
    raise ValueError(
        f"Unknown cache backend '{backend}'. Supported: local, redis."
    )
