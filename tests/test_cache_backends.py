"""Contract tests for CacheBackend implementations.

Each implementation gets the same test battery via parametrisation. Adding a
new backend = add it to `_BACKEND_FACTORIES`; the contract is enforced.
"""
from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio

from app.services.cache_backends import (
    CacheBackend,
    LocalCacheBackend,
    make_cache_backend,
)


def _local_factory(tmp_path) -> dict[str, Any]:
    return {
        "enabled": True,
        "directory": str(tmp_path / "cache"),
        "max_total_size_gb": 0.001,
        "eviction_policy": "lru",
    }


_BACKEND_FACTORIES: dict[str, Any] = {
    "local": (LocalCacheBackend, _local_factory),
}


@pytest_asyncio.fixture(params=list(_BACKEND_FACTORIES.keys()))
async def backend(request, tmp_path) -> CacheBackend:
    cls, cfg_factory = _BACKEND_FACTORIES[request.param]
    instance: CacheBackend = cls(cfg_factory(tmp_path))
    await instance.initialize()
    yield instance
    await instance.close()


@pytest.mark.asyncio
async def test_contract_initialize_marks_ready(backend: CacheBackend):
    assert backend.is_initialized() is True


@pytest.mark.asyncio
async def test_contract_close_marks_not_ready(backend: CacheBackend):
    await backend.close()
    assert backend.is_initialized() is False
    # After close, idempotent operations must not raise.
    assert await backend.get("anything") is None
    assert await backend.invalidate_key("anything") is False


@pytest.mark.asyncio
async def test_contract_put_then_get_round_trip(backend: CacheBackend):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    await backend.put("k1", b"payload-1", "model-x", cfg)
    assert await backend.get("k1") == b"payload-1"


@pytest.mark.asyncio
async def test_contract_get_miss_returns_none(backend: CacheBackend):
    assert await backend.get("does-not-exist") is None


@pytest.mark.asyncio
async def test_contract_invalidate_key(backend: CacheBackend):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    await backend.put("k", b"x", "model-y", cfg)
    assert await backend.invalidate_key("k") is True
    assert await backend.get("k") is None
    # Idempotent: deleting a deleted key returns False, never raises.
    assert await backend.invalidate_key("k") is False


@pytest.mark.asyncio
async def test_contract_invalidate_model_clears_only_that_model(backend: CacheBackend):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    await backend.put("a", b"1", "m1", cfg)
    await backend.put("b", b"2", "m1", cfg)
    await backend.put("c", b"3", "m2", cfg)

    deleted = await backend.invalidate_model("m1")
    assert deleted == 2
    assert await backend.get("a") is None
    assert await backend.get("b") is None
    assert await backend.get("c") == b"3"


@pytest.mark.asyncio
async def test_contract_invalidate_all(backend: CacheBackend):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    await backend.put("a", b"1", "m1", cfg)
    await backend.put("b", b"2", "m2", cfg)

    deleted = await backend.invalidate_all()
    assert deleted == 2
    assert await backend.get("a") is None


@pytest.mark.asyncio
async def test_contract_ttl_expiry(backend: CacheBackend):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10, "ttl_hours": 0}
    await backend.put("k", b"data", "m", cfg)
    # ttl_hours=0 → already expired on read.
    assert await backend.get("k") is None


@pytest.mark.asyncio
async def test_contract_invalidate_expired(backend: CacheBackend):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10, "ttl_hours": 0}
    for i in range(3):
        await backend.put(f"k{i}", b"x", "m", cfg)
    expired = await backend.invalidate_expired()
    assert expired == 3


@pytest.mark.asyncio
async def test_contract_record_miss_increments_stats(backend: CacheBackend):
    await backend.record_miss("m")
    await backend.record_miss("m")
    stats = await backend.stats("m")
    assert stats["miss_count"] == 2
    assert stats["hit_count"] == 0


@pytest.mark.asyncio
async def test_contract_hit_count_tracks_reads(backend: CacheBackend):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    await backend.put("k", b"data", "m", cfg)
    await backend.get("k")
    await backend.get("k")

    stats = await backend.stats("m")
    assert stats["hit_count"] == 2


@pytest.mark.asyncio
async def test_contract_stats_global_shape(backend: CacheBackend):
    cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
    await backend.put("k", b"data", "m", cfg)

    stats = await backend.stats()
    assert "global" in stats
    assert "per_model" in stats
    assert stats["global"]["total_entries"] == 1
    assert "m" in stats["per_model"]


# ── Factory ─────────────────────────────────────────────────────────


def test_factory_default_is_local(tmp_path):
    backend = make_cache_backend({
        "enabled": True,
        "directory": str(tmp_path / "cache"),
    })
    assert isinstance(backend, LocalCacheBackend)


def test_factory_explicit_local(tmp_path):
    backend = make_cache_backend({
        "enabled": True,
        "backend": "local",
        "directory": str(tmp_path / "cache"),
    })
    assert isinstance(backend, LocalCacheBackend)


def test_factory_unknown_backend_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown cache backend"):
        make_cache_backend({"backend": "telepathy", "directory": str(tmp_path)})


def test_factory_case_insensitive(tmp_path):
    backend = make_cache_backend({
        "backend": "LOCAL",
        "directory": str(tmp_path / "cache"),
    })
    assert isinstance(backend, LocalCacheBackend)


# ── CacheManager façade ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_manager_exposes_backend_property(tmp_path):
    from app.services.cache_manager import CacheManager

    mgr = CacheManager({
        "enabled": True,
        "directory": str(tmp_path / "cache"),
    })
    await mgr.initialize()
    try:
        assert isinstance(mgr.backend, LocalCacheBackend)
    finally:
        await mgr.close()


@pytest.mark.asyncio
async def test_cache_manager_accepts_injected_backend(tmp_path):
    """CacheManager(backend=...) must skip the factory and use the injected instance."""
    from app.services.cache_manager import CacheManager

    injected = LocalCacheBackend({
        "enabled": True,
        "directory": str(tmp_path / "injected"),
    })
    mgr = CacheManager(global_config={"enabled": True}, backend=injected)
    assert mgr.backend is injected

    await mgr.initialize()
    try:
        cfg = {"enabled": True, "strategy": "always", "max_size_mb": 10}
        await mgr.put("k", b"v", "m", cfg)
        assert await mgr.get("k") == b"v"
    finally:
        await mgr.close()
