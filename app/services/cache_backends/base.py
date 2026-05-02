from __future__ import annotations

from abc import ABC, abstractmethod


class CacheBackend(ABC):
    """Storage layer for cached responses — swappable behind CacheManager.

    Implementations must be safe to call concurrently from multiple coroutines.
    `cache_config` passed to put() is the per-model cache policy
    ({"strategy", "ttl_hours", "max_size_mb"}); backends use it to enforce
    per-model byte budgets and TTL.
    """

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def is_initialized(self) -> bool: ...

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Return cached bytes for `key` or None on miss/expiry."""

    @abstractmethod
    async def put(
        self, key: str, data: bytes, model_id: str, cache_config: dict
    ) -> None:
        """Store `data` under `key`, applying per-model and global byte budgets."""

    @abstractmethod
    async def record_miss(self, model_id: str) -> None:
        """Increment the miss counter for `model_id` (for hit-rate stats)."""

    @abstractmethod
    async def invalidate_key(self, key: str) -> bool: ...

    @abstractmethod
    async def invalidate_model(self, model_id: str) -> int: ...

    @abstractmethod
    async def invalidate_all(self) -> int: ...

    @abstractmethod
    async def invalidate_expired(self) -> int: ...

    @abstractmethod
    async def stats(self, model_id: str | None = None) -> dict: ...
