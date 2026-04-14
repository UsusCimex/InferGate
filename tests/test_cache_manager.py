"""Direct tests for CacheManager service."""
from __future__ import annotations

import pytest
import pytest_asyncio

from app.services.cache_manager import CacheManager, CacheStrategy


@pytest_asyncio.fixture
async def cache(tmp_path):
    mgr = CacheManager({
        "enabled": True,
        "directory": str(tmp_path / "cache"),
        "max_total_size_gb": 0.001,  # ~1MB
        "eviction_policy": "lru",
    })
    await mgr.initialize()
    yield mgr
    await mgr.close()


@pytest.mark.asyncio
async def test_put_and_get(cache):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    key = cache.make_key("model-a", {"prompt": "hello"})
    await cache.put(key, b"test data", "model-a", cfg)

    result = await cache.get(key)
    assert result == b"test data"


@pytest.mark.asyncio
async def test_get_nonexistent(cache):
    result = await cache.get("nonexistent-key")
    assert result is None


@pytest.mark.asyncio
async def test_ttl_expiration(cache):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10, "ttl_hours": 0}
    key = cache.make_key("model-a", {"prompt": "expire"})
    await cache.put(key, b"data", "model-a", cfg)

    # ttl_hours=0 means expires immediately
    result = await cache.get(key)
    assert result is None


@pytest.mark.asyncio
async def test_invalidate_key(cache):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    key = cache.make_key("model-a", {"prompt": "delete me"})
    await cache.put(key, b"data", "model-a", cfg)

    assert await cache.invalidate_key(key) is True
    assert await cache.get(key) is None
    assert await cache.invalidate_key(key) is False  # already deleted


@pytest.mark.asyncio
async def test_invalidate_model(cache):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    for i in range(3):
        key = cache.make_key("model-b", {"prompt": f"entry-{i}"})
        await cache.put(key, f"data-{i}".encode(), "model-b", cfg)

    count = await cache.invalidate_model("model-b")
    assert count == 3


@pytest.mark.asyncio
async def test_invalidate_all(cache):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    for model in ["m1", "m2"]:
        key = cache.make_key(model, {"prompt": "x"})
        await cache.put(key, b"data", model, cfg)

    count = await cache.invalidate_all()
    assert count == 2


@pytest.mark.asyncio
async def test_invalidate_expired(cache):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10, "ttl_hours": 0}
    for i in range(3):
        key = cache.make_key("model-x", {"prompt": f"exp-{i}"})
        await cache.put(key, b"data", "model-x", cfg)

    expired = await cache.invalidate_expired()
    assert expired == 3


@pytest.mark.asyncio
async def test_stats(cache):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    key = cache.make_key("model-s", {"prompt": "stats"})
    await cache.put(key, b"test", "model-s", cfg)
    await cache.get(key)  # hit

    stats = await cache.stats()
    assert stats["global"]["total_entries"] == 1
    assert "model-s" in stats["per_model"]


@pytest.mark.asyncio
async def test_stats_per_model(cache):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    key = cache.make_key("model-p", {"prompt": "per-model"})
    await cache.put(key, b"test", "model-p", cfg)

    stats = await cache.stats("model-p")
    assert stats["entries"] == 1


@pytest.mark.asyncio
async def test_should_cache_strategies():
    mgr = CacheManager({"enabled": True, "directory": "/tmp/test"})

    assert mgr.should_cache({"enabled": True, "strategy": "always"}, {}) is True
    assert mgr.should_cache({"enabled": True, "strategy": "never"}, {}) is False
    assert mgr.should_cache({"enabled": True, "strategy": "seed_only"}, {"seed": 42}) is True
    assert mgr.should_cache({"enabled": True, "strategy": "seed_only"}, {}) is False
    assert mgr.should_cache({"enabled": False, "strategy": "always"}, {}) is False


@pytest.mark.asyncio
async def test_should_cache_disabled():
    mgr = CacheManager({"enabled": False, "directory": "/tmp/test"})
    assert mgr.should_cache({"enabled": True, "strategy": "always"}, {}) is False


@pytest.mark.asyncio
async def test_record_miss(cache):
    await cache.record_miss("model-miss")
    await cache.record_miss("model-miss")

    stats = await cache.stats("model-miss")
    assert stats["miss_count"] >= 2


@pytest.mark.asyncio
async def test_make_key_deterministic(cache):
    key1 = cache.make_key("model", {"a": 1, "b": 2})
    key2 = cache.make_key("model", {"b": 2, "a": 1})
    assert key1 == key2  # sort_keys=True


@pytest.mark.asyncio
async def test_eviction_for_model(cache):
    # max_size_mb very small so eviction triggers
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 0.0005}
    big_data = b"x" * 400

    key1 = cache.make_key("evict-model", {"prompt": "first"})
    await cache.put(key1, big_data, "evict-model", cfg)

    key2 = cache.make_key("evict-model", {"prompt": "second"})
    await cache.put(key2, big_data, "evict-model", cfg)

    # Second entry stored; first may be evicted if over limit
    result2 = await cache.get(key2)
    assert result2 == big_data


@pytest.mark.asyncio
async def test_close_and_operations(cache):
    await cache.close()
    # Operations on closed cache should return safely
    assert await cache.get("any") is None
    assert await cache.invalidate_key("any") is False
    assert await cache.invalidate_model("any") == 0
    assert await cache.invalidate_all() == 0
    assert await cache.stats() == {}
