"""Background watcher for `config/models/*.yaml` — drives hot-reload.

Polls the directory every `interval` seconds and invokes `callback(config)`
for every file whose mtime advanced (creates + modifies). Removed files
log a warning — the corresponding model stays registered in the gateway
until the next restart. We defer auto-unregister because a file-system
race (editor saves via temp-file + rename) would otherwise briefly flash
the model as removed and disrupt inflight requests.

Why polling rather than watchdog / inotify:
  * ~17 YAML files × `os.stat()` per 2s is sub-millisecond CPU — not
    measurable in a gateway that already runs httpx/SQLite.
  * avoids a ~250KB dependency and its native-build requirement.
  * inotify is unreliable inside some WSL2 + Docker-on-Windows setups,
    which would undercut the DevEx value this feature exists for.

Safety properties:
  * YAML parse errors are logged and swallowed — an operator mid-edit
    must not take down the gateway.
  * A callback exception is caught per-file, so one bad reload doesn't
    skip the rest of a batch.
  * First scan establishes a baseline snapshot without firing callbacks —
    startup already loaded every file via `load_model_configs()`.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.config import ModelConfig
from app.config.loader import load_single_model_config

logger = logging.getLogger(__name__)

ReloadCallback = Callable[[ModelConfig], Awaitable[None]]


class ConfigWatcher:
    """Poll-based file watcher for per-model YAML configs.

    Lifecycle mirrors the other background services on the app:
        watcher = ConfigWatcher(dir, callback)
        watcher.start()   # schedules asyncio task
        ...
        watcher.stop()    # cancels and awaits
    """

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
        # path → last-observed mtime. First scan populates this without
        # dispatching callbacks — otherwise every startup would re-fire
        # reloads for every file.
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
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        try:
            while True:
                try:
                    await self.scan_once()
                except Exception as e:  # noqa: BLE001 — watcher loop must survive
                    logger.exception("Config watcher scan failed: %s", e)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            logger.info("Config watcher stopped")
            raise

    async def scan_once(self) -> list[Path]:
        """Single scan pass — returns the list of changed paths that
        were dispatched. Exposed for testing so we don't have to race
        the 2-second sleep loop."""
        if not self._models_dir.exists():
            return []
        current: dict[Path, float] = {}
        for path in self._models_dir.glob("*.yaml"):
            try:
                current[path] = path.stat().st_mtime
            except OSError as e:
                logger.debug("stat(%s) failed: %s", path, e)
                continue

        if not self._initialized:
            self._mtimes = current
            self._initialized = True
            return []

        changed: list[Path] = []
        for path, mtime in current.items():
            prev = self._mtimes.get(path)
            if prev is None:
                changed.append(path)  # newly appeared file
            elif mtime > prev + self._debounce:
                changed.append(path)  # genuine content edit

        for missing in set(self._mtimes) - set(current):
            logger.warning(
                "Config file removed: %s — model stays registered until restart",
                missing.name,
            )

        # Commit new mtimes even if a callback raises — otherwise we'd
        # retry the same broken file every 2 seconds forever.
        self._mtimes = current

        for path in changed:
            try:
                config = load_single_model_config(path)
            except Exception as e:  # noqa: BLE001 — parser errors are user-facing
                logger.error("Failed to parse %s: %s", path.name, e)
                continue
            try:
                await self._callback(config)
            except Exception as e:  # noqa: BLE001 — one bad reload shouldn't skip others
                logger.error("Reload callback for %s raised: %s", path.name, e)

        return changed
