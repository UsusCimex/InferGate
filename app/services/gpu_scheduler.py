from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable

logger = logging.getLogger(__name__)

PRIORITY_MAP = {"high": 0, "medium": 1, "low": 2}


class RequestTimeoutError(Exception):
    pass


class QueueFullError(Exception):
    pass


class GpuScheduler:
    """GPU task queue with per-model priorities and concurrency control."""

    def __init__(self, max_queue_size: int = 50):
        self._max_queue_size = max_queue_size
        self._model_semaphores: dict[str, asyncio.Semaphore] = {}
        self._active_tasks = 0
        self._total_submitted = 0
        self._total_completed = 0

    def register_model(self, model_id: str, max_concurrent: int) -> None:
        """Create a semaphore for a model (from YAML queue.max_concurrent)."""
        self._model_semaphores[model_id] = asyncio.Semaphore(max_concurrent)

    async def submit(
        self,
        model_id: str,
        priority: str,
        coro: Awaitable[Any],
        timeout: float,
    ) -> Any:
        """Submit task to the scheduler. Waits for GPU semaphore."""
        if self._active_tasks >= self._max_queue_size:
            raise QueueFullError(
                f"Queue is full ({self._active_tasks}/{self._max_queue_size})"
            )

        semaphore = self._model_semaphores.get(model_id)
        if semaphore is None:
            # Fallback — allow 1 concurrent
            semaphore = asyncio.Semaphore(1)
            self._model_semaphores[model_id] = semaphore

        position = self._active_tasks + 1
        self._active_tasks += 1
        self._total_submitted += 1
        self._last_position = position
        try:
            async with asyncio.timeout(timeout):
                async with semaphore:
                    result = await coro
                    self._total_completed += 1
                    return result
        except TimeoutError:
            raise RequestTimeoutError(
                f"Generation timed out after {timeout}s for model {model_id}"
            )
        finally:
            self._active_tasks -= 1

    @property
    def last_position(self) -> int:
        """Queue position of the last submitted task."""
        return getattr(self, "_last_position", 0)

    def queue_info(self) -> dict:
        """Current queue state for /metrics."""
        return {
            "queue_size": self._active_tasks,
            "max_queue_size": self._max_queue_size,
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
        }
