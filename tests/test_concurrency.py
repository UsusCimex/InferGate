"""Tests for concurrency, race conditions, and edge cases."""
from __future__ import annotations

import asyncio

import pytest

from app.services.gpu_scheduler import GpuScheduler, QueueFullError


@pytest.mark.asyncio
async def test_concurrent_scheduler_submissions():
    """Multiple concurrent submissions should not exceed queue size."""
    scheduler = GpuScheduler(max_queue_size=5)
    scheduler.register_model("test", 3)

    results = []

    async def slow_task(i):
        await asyncio.sleep(0.05)
        return i

    # Submit 5 tasks concurrently (at the limit)
    coros = [
        scheduler.submit("test", "medium", slow_task(i), timeout=5.0)
        for i in range(5)
    ]
    results = await asyncio.gather(*coros)
    assert len(results) == 5
    assert set(results) == {0, 1, 2, 3, 4}


@pytest.mark.asyncio
async def test_queue_full_error():
    """Exceeding queue size should raise QueueFullError."""
    scheduler = GpuScheduler(max_queue_size=2)
    scheduler.register_model("test", 1)

    barrier = asyncio.Event()

    async def blocking_task():
        await barrier.wait()
        return "done"

    # Fill the queue
    tasks = [
        asyncio.create_task(scheduler.submit("test", "medium", blocking_task(), timeout=5.0))
        for _ in range(2)
    ]
    await asyncio.sleep(0.01)  # let tasks start

    # Third submission should fail
    with pytest.raises(QueueFullError):
        await scheduler.submit("test", "medium", blocking_task(), timeout=1.0)

    barrier.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_scheduler_counter_consistency():
    """After all tasks complete, active_tasks should be 0."""
    scheduler = GpuScheduler(max_queue_size=10)
    scheduler.register_model("test", 5)

    async def noop():
        return True

    coros = [
        scheduler.submit("test", "medium", noop(), timeout=5.0)
        for _ in range(10)
    ]
    await asyncio.gather(*coros)

    info = scheduler.queue_info()
    assert info["queue_size"] == 0
    assert info["total_submitted"] == 10
    assert info["total_completed"] == 10


@pytest.mark.asyncio
async def test_concurrent_model_loads(services):
    """Loading the same model concurrently should not cause errors."""
    manager = services["manager"]

    async def load_text():
        return await manager.ensure_loaded("test-text")

    # 5 concurrent loads of the same model
    results = await asyncio.gather(*[load_text() for _ in range(5)])
    assert all(r.is_loaded() for r in results)


@pytest.mark.asyncio
async def test_concurrent_cache_operations(services):
    """Concurrent cache reads/writes should not corrupt data."""
    cache_mgr = services["cache"]
    cache_cfg = {"enabled": True, "strategy": "always", "max_size_mb": 100}

    async def write_and_read(i):
        key = f"test-key-{i}"
        data = f"data-{i}".encode()
        await cache_mgr.put(key, data, "test-model", cache_cfg)
        result = await cache_mgr.get(key)
        return result

    results = await asyncio.gather(*[write_and_read(i) for i in range(10)])
    for i, result in enumerate(results):
        assert result == f"data-{i}".encode()


@pytest.mark.asyncio
async def test_cache_corrupted_data(services):
    """Corrupted JSON in cache should not crash the endpoint."""
    cache_mgr = services["cache"]
    cache_cfg = {"enabled": True, "strategy": "always", "max_size_mb": 100}

    # Write invalid JSON
    await cache_mgr.put("bad-key", b"not valid json {{{", "test-model", cache_cfg)

    # Should be able to read it (it's raw bytes, not JSON)
    result = await cache_mgr.get("bad-key")
    assert result == b"not valid json {{{"


@pytest.mark.asyncio
async def test_cache_missing_file(services):
    """Cache entry with missing file should auto-invalidate."""
    cache_mgr = services["cache"]
    cache_cfg = {"enabled": True, "strategy": "always", "max_size_mb": 100}

    await cache_mgr.put("file-key", b"some data", "test-model", cache_cfg)

    # Verify it's cached
    assert await cache_mgr.get("file-key") == b"some data"

    # Delete the file manually (simulate corruption)
    from pathlib import Path
    async with cache_mgr._db.execute(
        "SELECT file_path FROM cache_entries WHERE key = ?", ("file-key",)
    ) as cursor:
        row = await cursor.fetchone()
    Path(row[0]).unlink()

    # Cache should return None and auto-clean the entry
    assert await cache_mgr.get("file-key") is None


@pytest.mark.asyncio
async def test_graceful_shutdown(services):
    """Shutdown should unload all models without errors."""
    manager = services["manager"]

    await manager.ensure_loaded("test-text")
    await manager.ensure_loaded("test-image")
    assert len(manager.loaded_models()) == 2

    await manager.shutdown()
    assert len(manager.loaded_models()) == 0
