from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.monitoring import record_embedding_batch

EmbedCall = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass
class _Batch:
    model_id: str
    run: EmbedCall
    inputs: list[str] = field(default_factory=list)
    waiters: list[tuple[int, int, asyncio.Future[list[list[float]]]]] = field(default_factory=list)
    timer: asyncio.TimerHandle | None = None


class EmbeddingBatcher:
    """Merges concurrent text-embedding requests with one key into one provider call.

    A batch runs when it holds `max_batch_size` inputs or `max_wait_ms` after its first request.
    """

    def __init__(self) -> None:
        self._open: dict[tuple[str, str], _Batch] = {}
        self._running: set[asyncio.Task] = set()

    async def embed(
        self,
        key: tuple[str, str],
        inputs: list[str],
        run: EmbedCall,
        max_batch_size: int,
        max_wait_ms: float,
    ) -> list[list[float]]:
        """Embed `inputs` in the open batch of `key` (model id and priority); `run` makes the call."""
        batch = self._open.get(key)
        if batch is not None and len(batch.inputs) + len(inputs) > max_batch_size:
            self._close(key, batch)
            batch = None
        loop = asyncio.get_running_loop()
        if batch is None:
            batch = self._open[key] = _Batch(model_id=key[0], run=run)
            batch.timer = loop.call_later(max_wait_ms / 1000, self._close, key, batch)

        start = len(batch.inputs)
        batch.inputs.extend(inputs)
        future: asyncio.Future[list[list[float]]] = loop.create_future()
        batch.waiters.append((start, len(batch.inputs), future))
        if len(batch.inputs) >= max_batch_size:
            self._close(key, batch)
        return await future

    def _close(self, key: tuple[str, str], batch: _Batch) -> None:
        if self._open.get(key) is not batch:
            return
        del self._open[key]
        if batch.timer is not None:
            batch.timer.cancel()
        task = asyncio.create_task(self._execute(batch))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    @staticmethod
    async def _execute(batch: _Batch) -> None:
        record_embedding_batch(batch.model_id, len(batch.inputs))
        try:
            vectors = await batch.run(batch.inputs)
            if len(vectors) != len(batch.inputs):
                raise RuntimeError(
                    f"{batch.model_id} returned {len(vectors)} vectors for {len(batch.inputs)} inputs"
                )
        except asyncio.CancelledError:
            for _, _, future in batch.waiters:
                future.cancel()
            raise
        except Exception as e:
            for _, _, future in batch.waiters:
                if not future.done():
                    future.set_exception(e)
            return
        for start, end, future in batch.waiters:
            if not future.done():
                future.set_result(vectors[start:end])
