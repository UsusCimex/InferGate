from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.config import ModelConfig
from app.config.loader import load_single_model_config

logger = logging.getLogger(__name__)

ReloadCallback = Callable[[ModelConfig], Awaitable[None]]


class ConfigWatcher:
    """Polls a directory of per-model YAML configs and fires a callback on change."""

    def __init__(
        self,
        models_dir: str | Path,
        callback: ReloadCallback,
        interval: float = 2.0,
        debounce: float = 0.5,
    ) -> None:
        self._models_dir = Path(models_dir)
        self._callback = callback
        self._interval = interval
        self._debounce = debounce
        self._task: asyncio.Task | None = None
        self._mtimes: dict[Path, float] = {}
        self._initialized = False

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())
        logger.info(
            "Config watcher started on %s (interval=%.1fs, debounce=%.1fs)",
            self._models_dir, self._interval, self._debounce,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        try:
            while True:
                try:
                    await self.scan_once()
                except Exception as e:
                    logger.exception("Config watcher scan failed: %s", e)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            logger.info("Config watcher stopped")
            raise

    async def scan_once(self) -> list[Path]:
        """Run one scan pass and fire the callback for each changed file."""
        if not self._models_dir.exists():
            return []
        current: dict[Path, float] = {}
        for path in self._models_dir.glob("*.yaml"):
            try:
                current[path] = path.stat().st_mtime
            except OSError as e:
                logger.debug("stat(%s) failed: %s", path, e)
                continue

        # First scan only snapshots — startup already loaded every file.
        if not self._initialized:
            self._mtimes = current
            self._initialized = True
            return []

        changed: list[Path] = []
        for path, mtime in current.items():
            prev = self._mtimes.get(path)
            if prev is None or mtime > prev + self._debounce:
                changed.append(path)

        for missing in set(self._mtimes) - set(current):
            logger.warning(
                "Config file removed: %s — model stays registered until restart",
                missing.name,
            )

        # Commit mtimes even on callback failure so a broken file isn't retried every tick.
        self._mtimes = current

        for path in changed:
            try:
                config = load_single_model_config(path)
            except Exception as e:
                logger.error("Failed to parse %s: %s", path.name, e)
                continue
            try:
                await self._callback(config)
            except Exception as e:
                logger.error("Reload callback for %s raised: %s", path.name, e)

        return changed
