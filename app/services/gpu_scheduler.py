from __future__ import annotations

import asyncio
import heapq
import logging
from collections.abc import Awaitable
from typing import Any

logger = logging.getLogger(__name__)


class RequestTimeoutError(Exception):
    pass


class QueueFullError(Exception):
    pass


# Lower number = higher priority (heap is min-heap).
_PRIORITY_VALUES = {"high": 0, "medium": 1, "low": 2}
_DEFAULT_PRIORITY = _PRIORITY_VALUES["medium"]


def _priority_value(priority: Any) -> int:
    """Normalise a string/enum priority into an integer sort key."""
    if priority is None:
        return _DEFAULT_PRIORITY
    value = priority.value if hasattr(priority, "value") else priority
    return _PRIORITY_VALUES.get(str(value).lower(), _DEFAULT_PRIORITY)


class _ModelQueue:
    """Per-model concurrency slot with a priority-ordered waiter heap."""

    __slots__ = ("active", "lock", "max_concurrent", "waiters")

    def __init__(self, max_concurrent: int) -> None:
        self.max_concurrent = max_concurrent
        self.active = 0
        # Heap of (priority_value, seq, future). Lower priority value runs first;
        # seq breaks ties so equal-priority requests are FIFO.
        self.waiters: list[tuple[int, int, asyncio.Future]] = []
        self.lock = asyncio.Lock()


class GpuScheduler:
    """GPU task queue with per-model priorities and concurrency control."""

    def __init__(self, max_queue_size: int = 50):
        self._max_queue_size = max_queue_size
        self._queues: dict[str, _ModelQueue] = {}
        self._active_tasks = 0
        self._total_submitted = 0
        self._total_completed = 0
        self._lock = asyncio.Lock()
        self._last_position = 0
        self._seq = 0

    def register_model(self, model_id: str, max_concurrent: int) -> None:
        """Create a queue slot for a model (from YAML queue.max_concurrent)."""
        self._queues[model_id] = _ModelQueue(max_concurrent)

    def update_concurrency(self, model_id: str, max_concurrent: int) -> None:
        """Update per-model concurrency limit. New limit applies at next slot acquire."""
        queue = self._queues.get(model_id)
        if queue is None:
            self.register_model(model_id, max_concurrent)
            return
        queue.max_concurrent = max_concurrent

    async def submit(
        self,
        model_id: str,
        priority: Any,
        coro: Awaitable[Any],
        timeout: float,  # noqa: ASYNC109 — SLO-level deadline, not a cancel token
    ) -> Any:
        """Submit task to the scheduler. Higher priority requests jump the line."""
        async with self._lock:
            if self._active_tasks >= self._max_queue_size:
                raise QueueFullError(
                    f"Queue is full ({self._active_tasks}/{self._max_queue_size})"
                )

            queue = self._queues.get(model_id)
            if queue is None:
                queue = _ModelQueue(1)
                self._queues[model_id] = queue

            self._active_tasks += 1
            self._total_submitted += 1
            self._last_position = self._active_tasks
            self._seq += 1
            seq = self._seq

        priority_val = _priority_value(priority)
        slot_held = False
        try:
            await self._acquire_slot(queue, priority_val, seq)
            slot_held = True
            try:
                async with asyncio.timeout(timeout):
                    result = await coro
                async with self._lock:
                    self._total_completed += 1
                return result
            except TimeoutError as e:
                raise RequestTimeoutError(
                    f"Generation timed out after {timeout}s for model {model_id}"
                ) from e
        finally:
            if slot_held:
                await self._release_slot(queue)
            async with self._lock:
                self._active_tasks -= 1

    async def _acquire_slot(
        self, queue: _ModelQueue, priority_val: int, seq: int
    ) -> None:
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        async with queue.lock:
            if queue.active < queue.max_concurrent and not queue.waiters:
                queue.active += 1
                return
            heapq.heappush(queue.waiters, (priority_val, seq, fut))

        try:
            await fut
        except asyncio.CancelledError:
            # Two cases: we were still waiting, or we were already granted a slot.
            async with queue.lock:
                for i, (_p, _s, f) in enumerate(queue.waiters):
                    if f is fut:
                        queue.waiters.pop(i)
                        heapq.heapify(queue.waiters)
                        raise
                # Waiter was granted a slot concurrently with cancellation —
                # hand it off to the next waiter so it is not wasted.
                self._handoff_locked(queue)
            raise

    async def _release_slot(self, queue: _ModelQueue) -> None:
        async with queue.lock:
            self._handoff_locked(queue)

    @staticmethod
    def _handoff_locked(queue: _ModelQueue) -> None:
        """Either wake the highest-priority waiter or decrement active count."""
        while queue.waiters:
            _, _, fut = heapq.heappop(queue.waiters)
            if not fut.done():
                fut.set_result(None)
                return
        queue.active -= 1

    @property
    def last_position(self) -> int:
        """Queue position of the last submitted task."""
        return self._last_position

    def queue_info(self) -> dict:
        """Current queue state for /metrics."""
        return {
            "queue_size": self._active_tasks,
            "max_queue_size": self._max_queue_size,
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
        }
