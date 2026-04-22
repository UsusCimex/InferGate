from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from collections import OrderedDict
from typing import Any

from app.config import ModelConfig
from app.providers.base import BaseProvider
from app.providers.registry import get_provider_class

logger = logging.getLogger(__name__)

_WORKER_MONITOR_INTERVAL = 10  # base seconds between health checks
_WORKER_MAX_BACKOFF = 300  # cap probing interval at 5 minutes


class ModelNotFoundError(Exception):
    pass


class WorkerNotReadyError(Exception):
    """Raised when a remote worker is not yet available."""
    pass


class ConfigError(ValueError):
    """Raised when the model/server configuration is internally inconsistent."""
    pass


class InsufficientResourcesError(Exception):
    """Raised when ensure_loaded cannot free enough VRAM for a new model —
    the byte-budget LRU tried to evict but every remaining loaded model
    is pinned, and the incoming model doesn't fit in the leftover budget.
    Router maps this to HTTP 503 with actionable copy."""


class ProviderManager:
    """Registry of providers. Loads configs, manages model lifecycle with LRU swapping.

    LRU uses two complementary caps:
    - `max_vram_budget_mb` (byte-budget): evicts until sum of declared
      vram_mb of loaded non-pinned models + new model's vram_mb
      ≤ budget. Disabled when set to 0.
    - `max_loaded_models` (counter): back-stop for models that didn't
      declare vram_mb or when budget is disabled.
    """

    def __init__(
        self,
        model_dir: str,
        max_loaded: int,
        pinned: list[str] | None = None,
        max_vram_budget_mb: int = 0,
        vram_headroom_mb: int = 0,
        category_reservations: dict[str, int] | None = None,
    ):
        self._registry: dict[str, BaseProvider] = {}
        self._loaded_order: OrderedDict[str, None] = OrderedDict()
        self._max_loaded = max_loaded
        self._max_vram_budget_mb = max_vram_budget_mb
        self._vram_headroom_mb = vram_headroom_mb
        self._model_dir = model_dir
        self._pinned = set(pinned or [])
        self._category_reservations = dict(category_reservations or {})
        self._state_lock = asyncio.Lock()
        self._model_locks: dict[str, asyncio.Lock] = {}
        self._monitor_task: asyncio.Task | None = None
        # LRU + watchdog skip models with active > 0 to avoid yanking
        # weights out from under an in-flight forward pass.
        self._active_counts: dict[str, int] = {}

    def validate_config(self) -> None:
        """Validate pinned models fit within the configured capacities.

        Raises ConfigError when pinned GPU models exceed either the count
        ceiling or the byte budget, which would leave no room for LRU swaps.
        """
        gpu_pinned = [m for m in self._pinned if self._is_gpu_model(m)]
        if len(gpu_pinned) >= self._max_loaded:
            raise ConfigError(
                f"{len(gpu_pinned)} pinned GPU models do not fit within "
                f"max_loaded_models={self._max_loaded}. Increase "
                f"gpu.max_loaded_models in server.yaml to at least "
                f"{len(gpu_pinned) + 1} or remove a pinned model."
            )
        if self._max_vram_budget_mb > 0:
            pinned_vram = sum(self._registry[m].vram_mb for m in gpu_pinned
                              if m in self._registry)
            effective_budget = self._max_vram_budget_mb - self._vram_headroom_mb
            if pinned_vram >= effective_budget:
                raise ConfigError(
                    f"Pinned GPU models declare {pinned_vram} MB VRAM, which "
                    f"meets or exceeds max_vram_budget_mb={self._max_vram_budget_mb} "
                    f"(minus headroom {self._vram_headroom_mb} MB). "
                    f"Increase the budget or drop a pinned model."
                )

    def discover_models(self, configs: list[ModelConfig]) -> None:
        """Register providers from model configs."""
        for config in configs:
            if not config.enabled:
                logger.info("Skipping disabled model: %s", config.id)
                continue

            # Resolve worker URL from env var (e.g. WORKER_URL_QWEN3_5_4B)
            if not config.worker_url:
                env_key = "WORKER_URL_" + re.sub(r"[^A-Z0-9]", "_", config.id.upper())
                env_url = os.environ.get(env_key)
                if env_url:
                    config.worker_url = env_url

            try:
                if config.worker_url:
                    provider = self._create_remote_provider(config)
                    logger.info("Registered remote model: %s -> %s", config.id, config.worker_url)
                else:
                    provider_cls = get_provider_class(config.provider_class)
                    provider = provider_cls(config)
                    logger.info("Registered model: %s (%s)", config.id, config.provider_class)
                self._registry[config.id] = provider
            except ValueError as e:
                logger.warning("Failed to register %s: %s", config.id, e)

    @staticmethod
    def _create_remote_provider(config: ModelConfig) -> BaseProvider:
        """Create a remote provider based on model category."""
        from app.providers.remote import (
            RemoteImageProvider,
            RemoteSttProvider,
            RemoteTextProvider,
            RemoteTtsProvider,
            RemoteUpscaleProvider,
        )

        category_map: dict[str, type[BaseProvider]] = {
            "text": RemoteTextProvider,
            "image": RemoteImageProvider,
            "tts": RemoteTtsProvider,
            "stt": RemoteSttProvider,
            "upscale": RemoteUpscaleProvider,
        }
        cls = category_map.get(config.category)
        if cls is None:
            raise ValueError(f"No remote provider for category '{config.category}'")
        return cls(config)

    def get(self, model_id: str) -> BaseProvider:
        """Get provider by ID."""
        if model_id not in self._registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        return self._registry[model_id]

    def get_config(self, model_id: str) -> ModelConfig:
        return self.get(model_id).config

    def _is_gpu_model(self, model_id: str) -> bool:
        """Check if model uses GPU (vram_mb > 0)."""
        provider = self._registry.get(model_id)
        return provider is not None and provider.vram_mb > 0

    def _is_remote(self, model_id: str) -> bool:
        """Check if model is served by a remote worker."""
        provider = self._registry.get(model_id)
        return provider is not None and bool(provider.config.worker_url)

    def _get_model_lock(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._model_locks:
            self._model_locks[model_id] = asyncio.Lock()
        return self._model_locks[model_id]

    # ── Worker monitor ────────────────────────────────────────────────

    def start_worker_monitor(self) -> None:
        """Start background task that connects to remote workers."""
        self._monitor_task = asyncio.create_task(self._monitor_workers())

    def stop_worker_monitor(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()

    async def _monitor_workers(self) -> None:
        """Periodically probe remote workers and connect when ready.

        Each worker gets an exponential-backoff retry schedule after
        consecutive failures, capped at _WORKER_MAX_BACKOFF seconds.
        Registry is re-read each iteration so reload_model-recreated
        provider instances are picked up without restarting the monitor.
        """
        next_probe: dict[str, float] = {}
        fail_counts: dict[str, int] = {}
        announced: set[str] = set()

        while True:
            now = asyncio.get_running_loop().time()
            remote_models = {
                mid: p for mid, p in self._registry.items() if p.config.worker_url
            }
            new_ids = set(remote_models) - announced
            if new_ids:
                logger.info("Worker monitor watching: %s", ", ".join(sorted(new_ids)))
                announced |= new_ids
            for mid in remote_models:
                next_probe.setdefault(mid, 0.0)
                fail_counts.setdefault(mid, 0)
            for mid in list(next_probe):
                if mid not in remote_models:
                    next_probe.pop(mid, None)
                    fail_counts.pop(mid, None)
                    announced.discard(mid)

            for model_id, provider in remote_models.items():
                if provider.is_loaded():
                    # Verify still healthy
                    if hasattr(provider, "check_health"):
                        healthy = await provider.check_health()
                        if not healthy:
                            logger.warning(
                                "Worker disconnected: %s (%s) — marking unavailable",
                                model_id, provider.config.worker_url,
                            )
                            provider._loaded = False
                            async with self._state_lock:
                                self._loaded_order.pop(model_id, None)
                            fail_counts[model_id] = 0
                            next_probe[model_id] = now
                    continue

                if now < next_probe[model_id]:
                    continue

                # Try to connect
                try:
                    await provider.load(self._model_dir)
                    async with self._state_lock:
                        self._loaded_order[model_id] = None
                    logger.info(
                        "Worker ready: %s (%s) — model is now available",
                        model_id, provider.config.worker_url,
                    )
                    fail_counts[model_id] = 0
                    next_probe[model_id] = now
                except Exception as e:
                    fail_counts[model_id] += 1
                    delay = min(
                        _WORKER_MONITOR_INTERVAL * (2 ** (fail_counts[model_id] - 1)),
                        _WORKER_MAX_BACKOFF,
                    )
                    next_probe[model_id] = now + delay
                    logger.debug(
                        "Worker %s (%s) not ready (fail #%d, retry in %ds): %s",
                        model_id, provider.config.worker_url,
                        fail_counts[model_id], int(delay), e,
                    )

            await asyncio.sleep(_WORKER_MONITOR_INTERVAL)

    # ── Model loading ─────────────────────────────────────────────────

    async def ensure_loaded(self, model_id: str) -> BaseProvider:
        """Load model if not loaded. LRU swap if no slots available."""
        # Fast path: already loaded — just touch LRU
        async with self._state_lock:
            provider = self.get(model_id)
            if provider.is_loaded():
                self._touch_lru(model_id)
                return provider

        # Remote models: fail fast — background monitor handles connection
        if self._is_remote(model_id):
            raise WorkerNotReadyError(
                f"Worker for model '{model_id}' is not available yet. "
                f"It may still be starting up — try again in a few seconds."
            )

        # Slow path: per-model lock so only one load at a time per model,
        # but other models remain accessible
        async with self._get_model_lock(model_id):
            # Re-check after acquiring lock (another request may have loaded it)
            if provider.is_loaded():
                async with self._state_lock:
                    self._touch_lru(model_id)
                return provider

            async with self._state_lock:
                if self._is_gpu_model(model_id):
                    await self._make_room(incoming_vram_mb=provider.vram_mb)

            await provider.load(self._model_dir)

            async with self._state_lock:
                self._loaded_order[model_id] = None
            return provider

    async def load_model(self, model_id: str) -> None:
        """Explicitly load a model."""
        await self.ensure_loaded(model_id)

    async def unload_model(self, model_id: str) -> None:
        """Explicitly unload a model."""
        async with self._get_model_lock(model_id):
            provider = self.get(model_id)
            if not provider.is_loaded():
                return
            await provider.unload()
            async with self._state_lock:
                self._loaded_order.pop(model_id, None)

    async def reload_model(self, config: ModelConfig) -> bool:
        """Re-register a model with a new config (hot-reload entry point).

        Returns True if a change was applied. Remote providers route through
        existing.reload() (POST /reload to worker) when worker_url is
        unchanged and the provider is connected; otherwise we rebuild.
        """
        model_id = config.id
        existing = self._registry.get(model_id)

        if not config.enabled:
            if existing is None:
                return False
            async with self._get_model_lock(model_id):
                if existing.is_loaded():
                    await existing.unload()
                async with self._state_lock:
                    self._loaded_order.pop(model_id, None)
                self._registry.pop(model_id, None)
            logger.info("Model %s disabled via config — unloaded & unregistered", model_id)
            return True

        if existing is not None and existing.config.model_dump() == config.model_dump():
            return False  # Identical save (editor touch, no content diff)

        # Resolve worker URL (same rule as discover_models) — in case the
        # YAML was edited and WORKER_URL_* env override still applies.
        if not config.worker_url:
            env_key = "WORKER_URL_" + re.sub(r"[^A-Z0-9]", "_", model_id.upper())
            env_url = os.environ.get(env_key)
            if env_url:
                config.worker_url = env_url

        # Remote hot-path: keep provider instance + httpx pool, push new
        # config to worker via /reload. Avoids disconnected-state flicker.
        if (
            existing is not None
            and existing.config.worker_url
            and config.worker_url == existing.config.worker_url
            and existing.is_loaded()
            and hasattr(existing, "reload")
        ):
            async with self._get_model_lock(model_id):
                try:
                    action = await existing.reload(config)  # type: ignore[attr-defined]
                except Exception as e:
                    # Worker reload failed — log and fall through to the
                    # recreate path, which at minimum updates gateway-side
                    # metadata so /v1/models reflects the new YAML.
                    logger.warning(
                        "Worker /reload for %s failed (%s) — falling back to local re-register",
                        model_id, e,
                    )
                else:
                    existing.config = config
                    logger.info(
                        "Reloaded model %s via worker /reload (action=%s)",
                        model_id, action,
                    )
                    return True

        async with self._get_model_lock(model_id):
            was_loaded = existing is not None and existing.is_loaded()
            if was_loaded:
                assert existing is not None
                try:
                    await existing.unload()
                except Exception as e:
                    logger.warning(
                        "Error unloading stale %s during reload — continuing: %s",
                        model_id, e,
                    )
                async with self._state_lock:
                    self._loaded_order.pop(model_id, None)

            try:
                if config.worker_url:
                    new_provider = self._create_remote_provider(config)
                else:
                    provider_cls = get_provider_class(config.provider_class)
                    new_provider = provider_cls(config)
            except ValueError as e:
                # Bad YAML (e.g. unknown provider_class) — keep the old
                # provider so a typo doesn't drop the model from registry.
                logger.error("reload_model(%s) rejected new config: %s", model_id, e)
                if existing is not None:
                    self._registry[model_id] = existing
                return False

            self._registry[model_id] = new_provider

            # Local providers: reload into GPU if previously loaded so the
            # user doesn't need to re-request. Remote ones reconnect via
            # worker_monitor.
            if was_loaded and not config.worker_url:
                await new_provider.load(self._model_dir)
                async with self._state_lock:
                    self._loaded_order[model_id] = None

        logger.info(
            "Reloaded model %s%s%s",
            model_id,
            " (new registration)" if existing is None else "",
            " — reloaded into GPU" if was_loaded and not config.worker_url else "",
        )
        return True

    def _loaded_vram_mb(self) -> int:
        """Sum of declared vram_mb over currently-loaded GPU models."""
        return sum(
            self._registry[m].vram_mb
            for m in self._loaded_order
            if self._is_gpu_model(m) and m in self._registry
        )

    async def _make_room(self, incoming_vram_mb: int = 0) -> None:
        """Evict LRU GPU models until both caps are satisfied:

        1. count: `len(loaded_gpu) < max_loaded_models` (existing guard)
        2. bytes: `loaded_vram + incoming_vram_mb <= max_vram_budget_mb
                   - vram_headroom_mb` (new, when budget > 0)

        Evicts the least-recently-used non-pinned model one at a time
        until both conditions hold. Bail out when all remaining loaded
        models are pinned — the new load will then likely fail at the
        driver level, but that's explicitly the operator's choice
        (they pinned too many).
        """
        effective_budget = (
            self._max_vram_budget_mb - self._vram_headroom_mb
            if self._max_vram_budget_mb > 0
            else 0
        )

        while True:
            gpu_loaded = [m for m in self._loaded_order if self._is_gpu_model(m)]
            count_ok = len(gpu_loaded) < self._max_loaded

            if effective_budget > 0:
                projected_vram = self._loaded_vram_mb() + max(incoming_vram_mb, 0)
                bytes_ok = projected_vram <= effective_budget
            else:
                bytes_ok = True

            if count_ok and bytes_ok:
                break

            victim_id = self._find_lru_victim()
            if victim_id is None:
                # Count-cap overflow with all-pinned is a soft warning
                # (driver-level OOM will likely catch). Byte-budget
                # overflow is harder — raise so the caller can 503 the
                # request instead of dying on CUDA OOM + swap-spiralling
                # the host.
                if not bytes_ok and effective_budget > 0:
                    raise InsufficientResourcesError(
                        f"No VRAM budget left for {incoming_vram_mb} MB — "
                        f"{self._loaded_vram_mb()} MB currently loaded, "
                        f"budget {effective_budget} MB, all loaded models pinned. "
                        f"Unpin a model or raise gpu.max_vram_budget_mb."
                    )
                logger.warning(
                    "Cannot make room (count=%d/%d) — all loaded models are pinned",
                    len(gpu_loaded), self._max_loaded,
                )
                break

            reason = "count" if not count_ok else "byte-budget"
            logger.info(
                "Evicting model %s (LRU, %s): freeing %d MB VRAM",
                victim_id, reason, self._registry[victim_id].vram_mb,
            )
            await self._registry[victim_id].unload()
            del self._loaded_order[victim_id]

    def _find_lru_victim(self) -> str | None:
        """Least-recently-used model eligible for eviction.

        Two passes: first honour category reservations, then fall back
        to violating them if the byte budget leaves no choice. Pinned
        and in-flight models are never touched."""
        for model_id in self._loaded_order:
            if not self._evictable(model_id):
                continue
            if self._would_violate_reservation(model_id):
                continue
            return model_id
        for model_id in self._loaded_order:
            if self._evictable(model_id):
                return model_id
        return None

    def _evictable(self, model_id: str) -> bool:
        if model_id in self._pinned:
            return False
        if self._active_counts.get(model_id, 0) > 0:
            return False
        return True

    def _would_violate_reservation(
        self, model_id: str, excluded: set[str] | None = None
    ) -> bool:
        """True if unloading `model_id` drops its category below the
        reserved floor. `excluded` is the set of models already counted
        as evicted in a simulated plan, so planning iterations see the
        correct post-eviction category counts."""
        provider = self._registry.get(model_id)
        if provider is None:
            return False
        category = provider.config.category
        reserved = self._category_reservations.get(category, 0)
        if reserved <= 0:
            return False
        excluded = excluded or set()
        remaining = 0
        for m in self._loaded_order:
            if m == model_id or m in excluded:
                continue
            p = self._registry.get(m)
            if p is not None and p.config.category == category:
                remaining += 1
        return remaining < reserved

    @contextlib.asynccontextmanager
    async def active_request(self, model_id: str):
        """Context manager that increments the active-request counter
        for `model_id` on enter and decrements on exit. Routers wrap
        `scheduler.submit(...)` with this so LRU eviction won't pick
        a model while it's serving a request.

        Count is updated under `_state_lock` so the watchdog + LRU
        observe a consistent view.
        """
        async with self._state_lock:
            self._active_counts[model_id] = self._active_counts.get(model_id, 0) + 1
        try:
            yield
        finally:
            async with self._state_lock:
                current = self._active_counts.get(model_id, 0)
                if current <= 1:
                    self._active_counts.pop(model_id, None)
                else:
                    self._active_counts[model_id] = current - 1

    def active_request_count(self, model_id: str) -> int:
        """Number of currently in-flight requests for this model. Safe
        to read without the state lock — dirty-read is fine for
        observability; eviction decisions always re-read under the lock."""
        return self._active_counts.get(model_id, 0)

    def _touch_lru(self, model_id: str) -> None:
        """Move model to end of LRU (most recently used). O(1) with OrderedDict."""
        if model_id in self._loaded_order:
            self._loaded_order.move_to_end(model_id)

    def list_models(self) -> list[dict[str, Any]]:
        """List all models with their status."""
        result = []
        for provider in self._registry.values():
            cfg = provider.config
            model_info: dict[str, Any] = {
                "id": cfg.id,
                "object": "model",
                "created": 0,
                "owned_by": "infergate",
                "display_name": cfg.display_name,
                "category": cfg.category,
                "loaded": provider.is_loaded(),
                "enabled": cfg.enabled,
                "metadata": cfg.metadata.model_dump(),
            }
            if cfg.worker_url:
                model_info["remote"] = True
                model_info["worker_url"] = cfg.worker_url
                model_info["worker_status"] = "connected" if provider.is_loaded() else "waiting"
            result.append(model_info)
        return result

    def loaded_models(self) -> list[str]:
        """Return list of currently loaded model IDs."""
        return list(self._loaded_order)

    async def shutdown(self, timeout_per_model: float = 30.0) -> None:
        """Unload all models on shutdown with per-model timeout.
        Remote workers are skipped — they manage their own lifecycle.
        """
        self.stop_worker_monitor()
        for model_id in list(self._loaded_order):
            if self._is_remote(model_id):
                continue
            try:
                async with asyncio.timeout(timeout_per_model):
                    await self._registry[model_id].unload()
            except TimeoutError:
                logger.warning("Timeout unloading %s after %.0fs", model_id, timeout_per_model)
            except Exception as e:
                logger.warning("Error unloading %s: %s", model_id, e)
        self._loaded_order.clear()
