from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.config import CacheStrategy
from app.services.cache_backends import CacheBackend, make_cache_backend

logger = logging.getLogger(__name__)

# Bump to invalidate entries produced by older code when the cache format changes.
CACHE_KEY_VERSION = "v1"


class CacheManager:
    """Cache façade: caching policy + key derivation; storage delegated to a CacheBackend.

    Routers stay backend-agnostic — swap LocalCacheBackend for Redis/S3 without touching them.
    """

    def __init__(
        self,
        global_config: dict[str, Any],
        backend: CacheBackend | None = None,
    ):
        self._enabled = global_config.get("enabled", True)
        self._backend: CacheBackend = backend or make_cache_backend(global_config)

    @property
    def backend(self) -> CacheBackend:
        return self._backend

    async def initialize(self) -> None:
        await self._backend.initialize()

    async def close(self) -> None:
        await self._backend.close()

    def is_initialized(self) -> bool:
        return self._backend.is_initialized()

    def should_cache(self, cache_config: dict, request_params: dict) -> bool:
        """Decide whether the response for `request_params` should be cached."""
        if not self._enabled:
            return False
        if not cache_config.get("enabled", False):
            return False
        strategy = CacheStrategy(cache_config.get("strategy", "never"))
        if strategy == CacheStrategy.NEVER:
            return False
        if strategy == CacheStrategy.ALWAYS:
            return True
        if strategy == CacheStrategy.SEED_ONLY:
            return "seed" in request_params
        return False

    def make_key(self, model_id: str, request_params: dict) -> str:
        """Build a deterministic cache key from `model_id` + `request_params`."""
        canonical = json.dumps(
            {"v": CACHE_KEY_VERSION, "model": model_id, **request_params},
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ── Storage delegation ─────────────────────────────────────────

    async def get(self, key: str) -> bytes | None:
        return await self._backend.get(key)

    async def put(
        self, key: str, data: bytes, model_id: str, cache_config: dict
    ) -> None:
        await self._backend.put(key, data, model_id, cache_config)

    async def record_miss(self, model_id: str) -> None:
        await self._backend.record_miss(model_id)

    async def invalidate_key(self, key: str) -> bool:
        return await self._backend.invalidate_key(key)

    async def invalidate_model(self, model_id: str) -> int:
        return await self._backend.invalidate_model(model_id)

    async def invalidate_all(self) -> int:
        return await self._backend.invalidate_all()

    async def invalidate_expired(self) -> int:
        return await self._backend.invalidate_expired()

    async def stats(self, model_id: str | None = None) -> dict:
        return await self._backend.stats(model_id)
