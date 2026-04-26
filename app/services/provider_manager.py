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

_WORKER_MONITOR_INTERVAL = 10
_WORKER_MAX_BACKOFF = 300


class ModelNotFoundError(Exception):
    pass


class WorkerNotReadyError(Exception):
    """Raised when a remote worker is not yet available."""


class ConfigError(ValueError):
    """Raised when the model/server configuration is internally inconsistent."""


class InsufficientResourcesError(Exception):
    """Raised when VRAM budget is exhausted and no evictable model is left."""


class ProviderManager:
    """Registry of model providers with VRAM-budget LRU swapping."""

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
        # Models with active_count > 0 must never be evicted mid-request.
        self._active_counts: dict[str, int] = {}

    def validate_config(self) -> None:
        """Raise ConfigError if pinned models don't leave room for LRU swaps."""
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
        """Build the RemoteProvider subclass matching `config.category`."""
        from app.providers.remote import (
            RemoteAudioEmbeddingProvider,
            RemoteImageProvider,
            RemoteSttProvider,
            RemoteTextEmbeddingProvider,
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
            "embedding-text": RemoteTextEmbeddingProvider,
            "embedding-audio": RemoteAudioEmbeddingProvider,
        }
        cls = category_map.get(config.category)
        if cls is None:
            raise ValueError(f"No remote provider for category '{config.category}'")
        return cls(config)

    def get(self, model_id: str) -> BaseProvider:
        """Return the provider registered for `model_id`."""
        if model_id not in self._registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        return self._registry[model_id]

    def get_config(self, model_id: str) -> ModelConfig:
        return self.get(model_id).config

    def _is_gpu_model(self, model_id: str) -> bool:
        provider = self._registry.get(model_id)
        return provider is not None and provider.vram_mb > 0

    def _is_remote(self, model_id: str) -> bool:
        provider = self._registry.get(model_id)
        return provider is not None and bool(provider.config.worker_url)

    def _get_model_lock(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._model_locks:
            self._model_locks[model_id] = asyncio.Lock()
        return self._model_locks[model_id]

    def start_worker_monitor(self) -> None:
        """Start the background task that tracks remote worker reachability."""
        self._monitor_task = asyncio.create_task(self._monitor_workers())

    def stop_worker_monitor(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()

    async def _monitor_workers(self) -> None:
        """Poll remote workers /health, updating loaded-state on disconnect."""
        next_probe: dict[str, float] = {}
        fail_counts: dict[str, int] = {}
        announced: set[str] = set()
        reachable: set[str] = set()

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
                    reachable.discard(mid)

            for model_id, provider in remote_models.items():
                if now < next_probe[model_id]:
                    continue

                healthy = False
                try:
                    if hasattr(provider, "check_health"):
                        healthy = await provider.check_health()
                except Exception:
                    healthy = False

                if healthy:
                    if model_id not in reachable:
                        logger.info(
                            "Worker reachable: %s (%s) — model loads on demand",
                            model_id, provider.config.worker_url,
                        )
                        reachable.add(model_id)
                    fail_counts[model_id] = 0
                    next_probe[model_id] = now + _WORKER_MONITOR_INTERVAL
                else:
                    # Drop loaded-state on disconnect so the planner doesn't reserve its VRAM.
                    if provider.is_loaded():
                        logger.warning(
                            "Worker disconnected: %s (%s) — marking unavailable",
                            model_id, provider.config.worker_url,
                        )
                        provider._loaded = False
                        async with self._state_lock:
                            self._loaded_order.pop(model_id, None)
                    reachable.discard(model_id)
                    fail_counts[model_id] += 1
                    delay = min(
                        _WORKER_MONITOR_INTERVAL * (2 ** (fail_counts[model_id] - 1)),
                        _WORKER_MAX_BACKOFF,
                    )
                    next_probe[model_id] = now + delay

            await asyncio.sleep(_WORKER_MONITOR_INTERVAL)

    async def ensure_loaded(self, model_id: str) -> BaseProvider:
        """Load `model_id` if not loaded (evicting LRU as needed) and return the provider."""
        async with self._state_lock:
            provider = self.get(model_id)
            if provider.is_loaded():
                self._touch_lru(model_id)
                return provider

        # Per-model lock — one concurrent load per model, but other models still accessible.
        async with self._get_model_lock(model_id):
            if provider.is_loaded():
                async with self._state_lock:
                    self._touch_lru(model_id)
                return provider

            async with self._state_lock:
                if self._is_gpu_model(model_id):
                    await self._make_room(incoming_vram_mb=provider.vram_mb)

            try:
                await provider.load(self._model_dir)
            except RuntimeError as e:
                if self._is_remote(model_id) and "not reachable" in str(e):
                    raise WorkerNotReadyError(
                        f"Worker for model '{model_id}' is not reachable. "
                        f"Check its container is running."
                    ) from e
                raise

            async with self._state_lock:
                self._loaded_order[model_id] = None
            return provider

    async def load_model(self, model_id: str) -> None:
        await self.ensure_loaded(model_id)

    async def unload_model(self, model_id: str) -> None:
        async with self._get_model_lock(model_id):
            provider = self.get(model_id)
            if not provider.is_loaded():
                return
            await provider.unload()
            async with self._state_lock:
                self._loaded_order.pop(model_id, None)

    async def reload_model(self, config: ModelConfig) -> bool:
        """Re-register a model with a new config; returns True when something changed."""
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
            # Editor touch with no content diff — no-op.
            return False

        if not config.worker_url:
            env_key = "WORKER_URL_" + re.sub(r"[^A-Z0-9]", "_", model_id.upper())
            env_url = os.environ.get(env_key)
            if env_url:
                config.worker_url = env_url

        # Remote hot-path: keep the httpx pool, push the new config via the worker's /reload.
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
                    # Fall through to recreate so gateway-side metadata still updates.
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
                # Keep the old provider on bad config so a typo doesn't drop the model.
                logger.error("reload_model(%s) rejected new config: %s", model_id, e)
                if existing is not None:
                    self._registry[model_id] = existing
                return False

            self._registry[model_id] = new_provider

            # Local reload only: remote reconnect goes through worker_monitor.
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
        return sum(
            self._registry[m].vram_mb
            for m in self._loaded_order
            if self._is_gpu_model(m) and m in self._registry
        )

    async def _make_room(self, incoming_vram_mb: int = 0) -> None:
        """Evict models to fit `incoming_vram_mb` in the VRAM budget."""
        plan = self._plan_eviction(incoming_vram_mb)
        if plan is None:
            effective_budget = self._effective_budget()
            if effective_budget > 0:
                raise InsufficientResourcesError(
                    f"Cannot fit {incoming_vram_mb} MB — "
                    f"{self._loaded_vram_mb()} MB loaded, budget {effective_budget} MB, "
                    f"remaining models all pinned or in-flight. "
                    f"Unpin a model or raise gpu.max_vram_budget_mb."
                )
            logger.warning(
                "Cannot make room — every loaded model is pinned or in-flight"
            )
            return
        for victim in plan:
            logger.info(
                "Evicting %s (freeing %d MB for incoming %d MB)",
                victim, self._registry[victim].vram_mb, incoming_vram_mb,
            )
            await self._registry[victim].unload()
            del self._loaded_order[victim]

    def _effective_budget(self) -> int:
        if self._max_vram_budget_mb <= 0:
            return 0
        return self._max_vram_budget_mb - self._vram_headroom_mb

    def _plan_eviction(self, incoming_vram_mb: int) -> list[str] | None:
        """Return the ordered LRU eviction plan, [] if it already fits, None if infeasible."""
        effective_budget = self._effective_budget()
        excluded: set[str] = set()
        plan: list[str] = []

        def fits() -> bool:
            gpu_loaded = [
                m for m in self._loaded_order
                if self._is_gpu_model(m) and m not in excluded
            ]
            if len(gpu_loaded) >= self._max_loaded:
                return False
            if effective_budget > 0:
                loaded_vram = sum(self._registry[m].vram_mb for m in gpu_loaded)
                if loaded_vram + max(incoming_vram_mb, 0) > effective_budget:
                    return False
            return True

        if fits():
            return []
        while True:
            victim = self._find_lru_victim(excluded=excluded)
            if victim is None:
                return None
            plan.append(victim)
            excluded.add(victim)
            if fits():
                return plan

    def _find_lru_victim(self, excluded: set[str] | None = None) -> str | None:
        """Return the LRU evictable model, first honouring reservations, then ignoring them."""
        excluded = excluded or set()
        for model_id in self._loaded_order:
            if model_id in excluded:
                continue
            if not self._evictable(model_id):
                continue
            if self._would_violate_reservation(model_id, excluded=excluded):
                continue
            return model_id
        for model_id in self._loaded_order:
            if model_id in excluded:
                continue
            if self._evictable(model_id):
                return model_id
        return None

    def _evictable(self, model_id: str) -> bool:
        if model_id in self._pinned:
            return False
        return self._active_counts.get(model_id, 0) == 0

    def _would_violate_reservation(
        self, model_id: str, excluded: set[str] | None = None
    ) -> bool:
        """True if unloading `model_id` would drop its category below the reserved floor."""
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
        """Bump the in-flight counter so LRU eviction skips this model during the request."""
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
        """Number of in-flight requests for this model (dirty-read, observability only)."""
        return self._active_counts.get(model_id, 0)

    def status_snapshot(self) -> dict[str, Any]:
        """Return a snapshot of LRU + budget state for operator dashboards."""
        loaded = [
            {
                "id": model_id,
                "category": self._registry[model_id].config.category,
                "vram_mb": self._registry[model_id].vram_mb,
                "pinned": model_id in self._pinned,
                "active_requests": self._active_counts.get(model_id, 0),
            }
            for model_id in self._loaded_order
        ]
        return {
            "loaded": loaded,
            "total_declared_vram_mb": self._loaded_vram_mb(),
            "max_loaded_models": self._max_loaded,
            "max_vram_budget_mb": self._max_vram_budget_mb,
            "vram_headroom_mb": self._vram_headroom_mb,
            "effective_budget_mb": self._effective_budget(),
            "pinned_models": sorted(self._pinned),
            "category_reservations": dict(self._category_reservations),
        }

    def preview_load(self, model_id: str) -> dict[str, Any]:
        """Dry-run `ensure_loaded(model_id)` — return the eviction plan without mutating state."""
        provider = self.get(model_id)
        incoming_mb = provider.vram_mb
        base: dict[str, Any] = {
            "model_id": model_id,
            "incoming_vram_mb": incoming_mb,
            "already_loaded": provider.is_loaded(),
        }
        if provider.is_loaded() or not self._is_gpu_model(model_id):
            return {**base, "feasible": True, "plan": [], "freed_mb": 0}
        plan = self._plan_eviction(incoming_mb)
        if plan is None:
            return {
                **base,
                "feasible": False,
                "plan": None,
                "reason": "no combination of evictions can free enough VRAM",
            }
        freed = sum(self._registry[m].vram_mb for m in plan)
        return {**base, "feasible": True, "plan": plan, "freed_mb": freed}

    def _touch_lru(self, model_id: str) -> None:
        if model_id in self._loaded_order:
            self._loaded_order.move_to_end(model_id)

    def list_models(self) -> list[dict[str, Any]]:
        """Return all registered models with lifecycle + metadata."""
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
        return list(self._loaded_order)

    async def shutdown(self, timeout_per_model: float = 30.0) -> None:
        """Unload every local model with a per-model timeout (remote workers manage their own)."""
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
