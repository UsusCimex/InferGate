from __future__ import annotations

import logging
import time
from typing import Any

from app.services.cache_backends.base import CacheBackend

try:
    import redis.asyncio as redis_async
except ImportError:  # pragma: no cover — optional dependency
    redis_async = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEFAULT_PREFIX = "infergate:cache"


def _decode(value: Any) -> Any:
    """Convert a single redis bytes value to str (other types pass through)."""
    if isinstance(value, bytes):
        return value.decode()
    return value


def _decode_keys(d: dict) -> dict:
    return {(_decode(k) if isinstance(k, bytes) else k): v for k, v in d.items()}


class RedisCacheBackend(CacheBackend):
    """Redis-backed CacheBackend — multi-instance safe, no filesystem dependency.

    Storage layout (all under `redis_prefix`, default "infergate:cache"):
      data:{key}        — bytes payload
      meta:{key}        — hash {model_id, size_bytes, created_at, last_accessed, ttl_expires?}
      hit:{key}         — counter (read on stats())
      lru               — sorted set, score=last_accessed, member=key (global LRU)
      lru:{model_id}    — sorted set per model
      miss:{model_id}   — counter
      models            — set of registered model_ids
    """

    def __init__(
        self,
        global_config: dict[str, Any],
        *,
        client: Any | None = None,
    ):
        if client is None and redis_async is None:
            raise RuntimeError(
                "RedisCacheBackend requires the `redis` package — install via "
                "`pip install redis`."
            )
        self._url: str = global_config.get("redis_url", "redis://localhost:6379/0")
        self._prefix: str = str(global_config.get("redis_prefix", _DEFAULT_PREFIX))
        self._max_total_bytes = int(global_config.get("max_total_size_gb", 10) * 1024**3)
        self._eviction_policy = global_config.get("eviction_policy", "lru")
        self._injected_client = client
        self._client: Any | None = client
        self._initialized = False

    # ── Key builders ───────────────────────────────────────────────

    def _data_key(self, key: str) -> str: return f"{self._prefix}:data:{key}"
    def _meta_key(self, key: str) -> str: return f"{self._prefix}:meta:{key}"
    def _hit_key(self, key: str) -> str: return f"{self._prefix}:hit:{key}"
    def _lru_key(self) -> str: return f"{self._prefix}:lru"
    def _model_lru_key(self, model_id: str) -> str: return f"{self._prefix}:lru:{model_id}"
    def _miss_key(self, model_id: str) -> str: return f"{self._prefix}:miss:{model_id}"
    def _models_key(self) -> str: return f"{self._prefix}:models"

    # ── Lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        if self._client is None:
            assert redis_async is not None
            self._client = redis_async.from_url(self._url)
        # Probe — fail fast if Redis unreachable.
        await self._client.ping()
        self._initialized = True

    async def close(self) -> None:
        if self._client is not None and self._injected_client is None:
            # Only close clients we created — injected clients (tests) belong to caller.
            await self._client.aclose()
        self._client = None
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    # ── Public API ─────────────────────────────────────────────────

    async def record_miss(self, model_id: str) -> None:
        if not self._initialized:
            return
        await self._client.incr(self._miss_key(model_id))
        await self._client.sadd(self._models_key(), model_id)

    async def get(self, key: str) -> bytes | None:
        if not self._initialized:
            return None
        meta_raw = await self._client.hgetall(self._meta_key(key))
        if not meta_raw:
            return None
        meta = _decode_keys(meta_raw)

        ttl_raw = meta.get("ttl_expires")
        if ttl_raw:
            ttl_v = float(_decode(ttl_raw))
            if time.time() > ttl_v:
                await self.invalidate_key(key)
                return None

        data = await self._client.get(self._data_key(key))
        if data is None:
            await self.invalidate_key(key)
            return None

        model_id = _decode(meta.get("model_id"))
        now = time.time()
        await self._client.zadd(self._lru_key(), {key: now})
        if model_id:
            await self._client.zadd(self._model_lru_key(model_id), {key: now})
        await self._client.hset(self._meta_key(key), "last_accessed", now)
        await self._client.incr(self._hit_key(key))
        return data

    async def put(
        self, key: str, data: bytes, model_id: str, cache_config: dict
    ) -> None:
        if not self._initialized:
            return

        size_bytes = len(data)

        max_model = int(cache_config.get("max_size_mb", 0)) * 1024 * 1024
        if max_model > 0:
            await self._evict_for_model(model_id, max_model, size_bytes)
        await self._evict_global(size_bytes)

        ttl_hours = cache_config.get("ttl_hours")
        now = time.time()
        meta: dict[str, Any] = {
            "model_id": model_id,
            "size_bytes": size_bytes,
            "created_at": now,
            "last_accessed": now,
        }
        if ttl_hours is not None:
            meta["ttl_expires"] = now + ttl_hours * 3600

        await self._client.set(self._data_key(key), data)
        await self._client.hset(self._meta_key(key), mapping=meta)
        await self._client.zadd(self._lru_key(), {key: now})
        await self._client.zadd(self._model_lru_key(model_id), {key: now})
        await self._client.sadd(self._models_key(), model_id)
        # Reset hit counter for the new entry (INSERT OR REPLACE semantics).
        await self._client.delete(self._hit_key(key))

    async def invalidate_key(self, key: str) -> bool:
        if not self._initialized:
            return False
        meta_raw = await self._client.hgetall(self._meta_key(key))
        if not meta_raw:
            return False
        meta = _decode_keys(meta_raw)
        model_id = _decode(meta.get("model_id"))

        await self._client.delete(self._data_key(key))
        await self._client.delete(self._meta_key(key))
        await self._client.delete(self._hit_key(key))
        await self._client.zrem(self._lru_key(), key)
        if model_id:
            await self._client.zrem(self._model_lru_key(model_id), key)
        return True

    async def invalidate_model(self, model_id: str) -> int:
        if not self._initialized:
            return 0
        members = await self._client.zrange(self._model_lru_key(model_id), 0, -1)
        keys = [_decode(m) for m in members]
        for k in keys:
            await self.invalidate_key(k)
        await self._client.delete(self._model_lru_key(model_id))
        await self._client.delete(self._miss_key(model_id))
        await self._client.srem(self._models_key(), model_id)
        return len(keys)

    async def invalidate_all(self) -> int:
        if not self._initialized:
            return 0
        members = await self._client.zrange(self._lru_key(), 0, -1)
        keys = [_decode(m) for m in members]
        for k in keys:
            await self.invalidate_key(k)
        # Per-model state — mirror the Local backend's scope.
        models = await self._client.smembers(self._models_key())
        for m in models:
            mid = _decode(m)
            await self._client.delete(self._model_lru_key(mid))
            await self._client.delete(self._miss_key(mid))
        await self._client.delete(self._models_key())
        await self._client.delete(self._lru_key())
        return len(keys)

    async def invalidate_expired(self) -> int:
        if not self._initialized:
            return 0
        now = time.time()
        members = await self._client.zrange(self._lru_key(), 0, -1)
        keys = [_decode(m) for m in members]
        expired: list[str] = []
        for k in keys:
            ttl_raw = await self._client.hget(self._meta_key(k), "ttl_expires")
            if ttl_raw is None:
                continue
            if float(_decode(ttl_raw)) < now:
                expired.append(k)
        for k in expired:
            await self.invalidate_key(k)
        return len(expired)

    async def stats(self, model_id: str | None = None) -> dict:
        if not self._initialized:
            return {}

        if model_id:
            return await self._model_stats(model_id)

        members = await self._client.zrange(self._lru_key(), 0, -1)
        keys = [_decode(m) for m in members]
        total_entries = len(keys)
        total_bytes = 0
        models_seen: set[str] = set()
        for k in keys:
            sz = await self._client.hget(self._meta_key(k), "size_bytes")
            if sz is not None:
                total_bytes += int(_decode(sz))
            mid = await self._client.hget(self._meta_key(k), "model_id")
            if mid is not None:
                models_seen.add(_decode(mid))

        per_model: dict[str, dict] = {}
        for m in sorted(models_seen):
            per_model[m] = await self._model_stats(m)

        return {
            "global": {
                "total_entries": total_entries,
                "total_size_mb": round(total_bytes / (1024 * 1024), 1),
                "max_size_mb": round(self._max_total_bytes / (1024 * 1024), 1),
                "eviction_policy": self._eviction_policy,
            },
            "per_model": per_model,
        }

    async def _model_stats(self, model_id: str) -> dict:
        members = await self._client.zrange(self._model_lru_key(model_id), 0, -1)
        keys = [_decode(m) for m in members]
        entries = len(keys)
        size_bytes = 0
        hits = 0
        for k in keys:
            sz = await self._client.hget(self._meta_key(k), "size_bytes")
            if sz is not None:
                size_bytes += int(_decode(sz))
            h = await self._client.get(self._hit_key(k))
            if h is not None:
                hits += int(_decode(h))

        misses_raw = await self._client.get(self._miss_key(model_id))
        misses = int(_decode(misses_raw)) if misses_raw is not None else 0

        total = hits + misses
        hit_rate = round(hits / total * 100, 1) if total > 0 else 0.0

        return {
            "entries": entries,
            "size_mb": round(size_bytes / (1024 * 1024), 1),
            "hit_count": hits,
            "miss_count": misses,
            "hit_rate_percent": hit_rate,
        }

    # ── Eviction ───────────────────────────────────────────────────

    async def _evict_for_model(
        self, model_id: str, max_bytes: int, needed: int
    ) -> None:
        keys = [_decode(m) for m in await self._client.zrange(
            self._model_lru_key(model_id), 0, -1
        )]
        current = 0
        sizes: dict[str, int] = {}
        for k in keys:
            sz = await self._client.hget(self._meta_key(k), "size_bytes")
            if sz is not None:
                sizes[k] = int(_decode(sz))
                current += sizes[k]

        while current + needed > max_bytes and keys:
            oldest = keys.pop(0)
            current -= sizes.get(oldest, 0)
            await self.invalidate_key(oldest)

    async def _evict_global(self, needed: int) -> None:
        keys = [_decode(m) for m in await self._client.zrange(
            self._lru_key(), 0, -1
        )]
        current = 0
        sizes: dict[str, int] = {}
        for k in keys:
            sz = await self._client.hget(self._meta_key(k), "size_bytes")
            if sz is not None:
                sizes[k] = int(_decode(sz))
                current += sizes[k]

        while current + needed > self._max_total_bytes and keys:
            oldest = keys.pop(0)
            current -= sizes.get(oldest, 0)
            await self.invalidate_key(oldest)
