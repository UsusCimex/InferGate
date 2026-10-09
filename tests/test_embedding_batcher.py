from __future__ import annotations

import asyncio

import pytest

from app.services.embedding_batcher import EmbeddingBatcher


class _Recorder:
    """Embedding call that records each batch and encodes a string as [len(s)]."""

    def __init__(self, fail: bool = False) -> None:
        self.batches: list[list[str]] = []
        self._fail = fail

    async def __call__(self, inputs: list[str]) -> list[list[float]]:
        self.batches.append(list(inputs))
        if self._fail:
            raise ValueError("bad batch")
        return [[float(len(s))] for s in inputs]


async def test_concurrent_requests_share_one_call():
    batcher, run = EmbeddingBatcher(), _Recorder()
    results = await asyncio.gather(
        batcher.embed(("m", "medium"), ["a"], run, 32, 20),
        batcher.embed(("m", "medium"), ["bb", "ccc"], run, 32, 20),
        batcher.embed(("m", "medium"), ["dddd"], run, 32, 20),
    )
    assert run.batches == [["a", "bb", "ccc", "dddd"]]
    assert results == [[[1.0]], [[2.0], [3.0]], [[4.0]]]


async def test_full_batch_runs_without_waiting_for_the_window():
    batcher, run = EmbeddingBatcher(), _Recorder()
    results = await asyncio.wait_for(
        asyncio.gather(*(batcher.embed(("m", "medium"), [s], run, 3, 60_000) for s in "xyz")),
        timeout=5,
    )
    assert run.batches == [["x", "y", "z"]]
    assert results == [[[1.0]]] * 3


async def test_request_that_overflows_the_batch_starts_a_new_one():
    batcher, run = EmbeddingBatcher(), _Recorder()
    await asyncio.gather(
        batcher.embed(("m", "medium"), ["a", "b"], run, 3, 20),
        batcher.embed(("m", "medium"), ["c", "d"], run, 3, 20),
    )
    assert run.batches == [["a", "b"], ["c", "d"]]


async def test_keys_do_not_mix():
    batcher, run = EmbeddingBatcher(), _Recorder()
    await asyncio.gather(
        batcher.embed(("m", "high"), ["a"], run, 32, 20),
        batcher.embed(("m", "low"), ["b"], run, 32, 20),
        batcher.embed(("other", "high"), ["c"], run, 32, 20),
    )
    assert sorted(run.batches) == [["a"], ["b"], ["c"]]


async def test_failed_call_reaches_every_request():
    batcher, run = EmbeddingBatcher(), _Recorder(fail=True)
    results = await asyncio.gather(
        batcher.embed(("m", "medium"), ["a"], run, 32, 20),
        batcher.embed(("m", "medium"), ["b"], run, 32, 20),
        return_exceptions=True,
    )
    assert run.batches == [["a", "b"]]
    assert all(isinstance(r, ValueError) for r in results)


async def test_cancelled_request_leaves_the_rest_of_the_batch():
    batcher, run = EmbeddingBatcher(), _Recorder()
    gone = asyncio.create_task(batcher.embed(("m", "medium"), ["a"], run, 32, 20))
    kept = asyncio.create_task(batcher.embed(("m", "medium"), ["bb"], run, 32, 20))
    await asyncio.sleep(0)
    gone.cancel()
    assert await kept == [[2.0]]
    with pytest.raises(asyncio.CancelledError):
        await gone
    assert run.batches == [["a", "bb"]]
