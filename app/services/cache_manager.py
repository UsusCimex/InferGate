from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    ALWAYS = "always"
    SEED_ONLY = "seed_only"
    NEVER = "never"


class CacheManager:
    """Manages caching with per-model settings. Uses filesystem + SQLite metadata."""

    def __init__(self, global_config: dict[str, Any]):
        self._base_dir = Path(global_config.get("directory", "./cache"))
        self._max_total_bytes = int(global_config.get("max_total_size_gb", 10) * 1024**3)
        self._eviction_policy = global_config.get("eviction_policy", "lru")
        self._enabled = global_config.get("enabled", True)
        self._db_path = self._base_dir / "_meta.db"
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create cache directory and initialize SQLite metadata DB."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                ttl_expires REAL,
                hit_count INTEGER DEFAULT 0
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_model ON cache_entries(model_id)"
        )
        # Track cache misses separately for accurate hit rate
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache_stats (
                model_id TEXT PRIMARY KEY,
                miss_count INTEGER DEFAULT 0
            )
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def record_miss(self, model_id: str) -> None:
        """Record a cache miss for accurate hit rate tracking."""
        if not self._db:
            return
        await self._db.execute(
            """INSERT INTO cache_stats (model_id, miss_count) VALUES (?, 1)
               ON CONFLICT(model_id) DO UPDATE SET miss_count = miss_count + 1""",
            (model_id,),
        )
        await self._db.commit()

    def should_cache(self, cache_config: dict, request_params: dict) -> bool:
        """Determine whether to cache based on model's cache strategy."""
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
        """Create deterministic cache key from model + params."""
        canonical = json.dumps({"model": model_id, **request_params}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def get(self, key: str) -> bytes | None:
        """Get from cache. Updates LRU stats on hit."""
        if not self._db:
            return None
        async with self._db.execute(
            "SELECT file_path, ttl_expires FROM cache_entries WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None

        file_path, ttl_expires = row

        # Check TTL
        if ttl_expires and time.time() > ttl_expires:
            await self.invalidate_key(key)
            return None

        path = Path(file_path)
        if not path.exists():
            await self.invalidate_key(key)
            return None

        # Update access time and hit count
        await self._db.execute(
            "UPDATE cache_entries SET last_accessed = ?, hit_count = hit_count + 1 WHERE key = ?",
            (time.time(), key),
        )
        await self._db.commit()
        return path.read_bytes()

    async def put(
        self, key: str, data: bytes, model_id: str, cache_config: dict
    ) -> None:
        """Store in cache. Respects per-model and global limits."""
        if not self._db:
            return

        model_dir = self._base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Use first 2 chars of key as subdirectory for sharding
        shard_dir = model_dir / key[:2]
        shard_dir.mkdir(exist_ok=True)

        # Determine file extension from data
        ext = _guess_extension(data)
        file_path = shard_dir / f"{key}{ext}"

        size_bytes = len(data)

        # Evict if needed
        max_model_bytes = int(cache_config.get("max_size_mb", 0)) * 1024 * 1024
        if max_model_bytes > 0:
            await self._evict_for_model(model_id, max_model_bytes, size_bytes)
        await self._evict_global(size_bytes)

        ttl_hours = cache_config.get("ttl_hours")
        ttl_expires = time.time() + ttl_hours * 3600 if ttl_hours is not None else None

        # Write to temp file first, then commit DB, then atomic rename
        tmp_path = file_path.with_suffix(".tmp")
        try:
            tmp_path.write_bytes(data)
            await self._db.execute(
                """INSERT OR REPLACE INTO cache_entries
                   (key, model_id, file_path, size_bytes, created_at, last_accessed, ttl_expires, hit_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                (key, model_id, str(file_path), size_bytes, time.time(), time.time(), ttl_expires),
            )
            await self._db.commit()
            tmp_path.replace(file_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    async def invalidate_key(self, key: str) -> bool:
        if not self._db:
            return False
        async with self._db.execute(
            "SELECT file_path FROM cache_entries WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return False
        path = Path(row[0])
        if path.exists():
            path.unlink()
        await self._db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        await self._db.commit()
        return True

    async def invalidate_model(self, model_id: str) -> int:
        if not self._db:
            return 0
        async with self._db.execute(
            "SELECT COUNT(*) FROM cache_entries WHERE model_id = ?", (model_id,)
        ) as cursor:
            row = await cursor.fetchone()
            count = row[0] if row else 0

        await self._db.execute("DELETE FROM cache_entries WHERE model_id = ?", (model_id,))
        await self._db.commit()

        model_dir = self._base_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        return count

    async def invalidate_all(self) -> int:
        if not self._db:
            return 0
        async with self._db.execute("SELECT COUNT(*) FROM cache_entries") as cursor:
            row = await cursor.fetchone()
            count = row[0] if row else 0

        await self._db.execute("DELETE FROM cache_entries")
        await self._db.commit()

        # Remove all model subdirectories but keep SQLite files
        for child in self._base_dir.iterdir():
            if child.name.startswith("_meta.db"):
                continue  # skip _meta.db, _meta.db-wal, _meta.db-shm
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            elif child.is_file():
                child.unlink(missing_ok=True)
        return count

    async def invalidate_expired(self) -> int:
        if not self._db:
            return 0
        now = time.time()
        async with self._db.execute(
            "SELECT key, file_path FROM cache_entries WHERE ttl_expires IS NOT NULL AND ttl_expires < ?",
            (now,),
        ) as cursor:
            rows = await cursor.fetchall()
        for key, file_path in rows:
            path = Path(file_path)
            if path.exists():
                path.unlink(missing_ok=True)
        await self._db.execute(
            "DELETE FROM cache_entries WHERE ttl_expires IS NOT NULL AND ttl_expires < ?",
            (now,),
        )
        await self._db.commit()
        return len(rows)

    async def stats(self, model_id: str | None = None) -> dict:
        """Cache statistics (global or per-model)."""
        if not self._db:
            return {}

        if model_id:
            return await self._model_stats(model_id)

        # Global stats
        async with self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM cache_entries"
        ) as cursor:
            row = await cursor.fetchone()
            total_entries, total_bytes = row if row else (0, 0)

        # Per-model stats
        per_model = {}
        async with self._db.execute(
            "SELECT DISTINCT model_id FROM cache_entries"
        ) as cursor:
            model_ids = [row[0] async for row in cursor]

        for mid in model_ids:
            per_model[mid] = await self._model_stats(mid)

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
        if not self._db:
            return {}
        async with self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0), COALESCE(SUM(hit_count), 0) FROM cache_entries WHERE model_id = ?",
            (model_id,),
        ) as cursor:
            row = await cursor.fetchone()
            entries, size_bytes, hits = row if row else (0, 0, 0)

        async with self._db.execute(
            "SELECT COALESCE(miss_count, 0) FROM cache_stats WHERE model_id = ?",
            (model_id,),
        ) as cursor:
            miss_row = await cursor.fetchone()
            misses = miss_row[0] if miss_row else entries

        hit_rate = round(hits / (hits + misses) * 100, 1) if (hits + misses) > 0 else 0.0

        return {
            "entries": entries,
            "size_mb": round(size_bytes / (1024 * 1024), 1),
            "hit_count": hits,
            "miss_count": misses,
            "hit_rate_percent": hit_rate,
        }

    async def _evict_for_model(self, model_id: str, max_bytes: int, needed: int) -> None:
        """Evict LRU entries for a specific model to fit within its limit."""
        async with self._db.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries WHERE model_id = ?",
            (model_id,),
        ) as cursor:
            row = await cursor.fetchone()
            current = row[0] if row else 0

        while current + needed > max_bytes:
            async with self._db.execute(
                "SELECT key, file_path, size_bytes FROM cache_entries WHERE model_id = ? ORDER BY last_accessed ASC LIMIT 1",
                (model_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                break
            key, file_path, size = row
            Path(file_path).unlink(missing_ok=True)
            await self._db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            current -= size
        await self._db.commit()

    async def _evict_global(self, needed: int) -> None:
        """Evict LRU entries globally to fit within total limit."""
        async with self._db.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries"
        ) as cursor:
            row = await cursor.fetchone()
            current = row[0] if row else 0

        while current + needed > self._max_total_bytes:
            async with self._db.execute(
                "SELECT key, file_path, size_bytes FROM cache_entries ORDER BY last_accessed ASC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                break
            key, file_path, size = row
            Path(file_path).unlink(missing_ok=True)
            await self._db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            current -= size
        await self._db.commit()


def _guess_extension(data: bytes) -> str:
    """Guess file extension from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        return ".mp3"
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    return ".bin"
