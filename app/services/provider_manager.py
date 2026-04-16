from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import OrderedDict
from typing import Any

from app.config import ModelConfig
from app.providers.base import BaseProvider
from app.providers.registry import get_provider_class

logger = logging.getLogger(__name__)

_WORKER_MONITOR_INTERVAL = 10  # seconds between health checks


class ModelNotFoundError(Exception):
    pass


class WorkerNotReadyError(Exception):
    """Raised when a remote worker is not yet available."""
    pass


class ProviderManager:
    """Registry of providers. Loads configs, manages model lifecycle with LRU swapping."""

    def __init__(self, model_dir: str, max_loaded: int, pinned: list[str] | None = None):
        self._registry: dict[str, BaseProvider] = {}
        self._loaded_order: OrderedDict[str, None] = OrderedDict()
        self._max_loaded = max_loaded
        self._model_dir = model_dir
        self._pinned = set(pinned or [])
        self._state_lock = asyncio.Lock()  # protects _loaded_order and state changes
        self._model_locks: dict[str, asyncio.Lock] = {}  # per-model load serialization
        self._monitor_task: asyncio.Task | None = None

    def validate_config(self) -> None:
        """Validate pinned models vs max_loaded capacity. Auto-correct if needed."""
        gpu_pinned = [m for m in self._pinned if self._is_gpu_model(m)]
        if len(gpu_pinned) >= self._max_loaded:
            old_max = self._max_loaded
            self._max_loaded = len(gpu_pinned) + 1
            logger.warning(
                "Pinned GPU models (%d) >= max_loaded_models (%d). "
                "Auto-increased max_loaded_models to %d.",
                len(gpu_pinned),
                old_max,
                self._max_loaded,
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
        from app.providers.remote import RemoteImageProvider, RemoteTextProvider, RemoteTtsProvider

        category_map: dict[str, type[BaseProvider]] = {
            "text": RemoteTextProvider,
            "image": RemoteImageProvider,
            "tts": RemoteTtsProvider,
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
        """Periodically probe remote workers and connect when ready."""
        remote_models = {
            mid: p for mid, p in self._registry.items() if p.config.worker_url
        }
        if remote_models:
            waiting = ", ".join(remote_models.keys())
            logger.info("Worker monitor started — watching %d workers: %s", len(remote_models), waiting)

        while True:
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
                except Exception:
                    pass  # Will retry next cycle

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
                    await self._make_room()

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

    async def _make_room(self) -> None:
        """Unload LRU GPU models until there's room."""
        while True:
            gpu_loaded = [m for m in self._loaded_order if self._is_gpu_model(m)]
            if len(gpu_loaded) < self._max_loaded:
                break
            victim_id = self._find_lru_victim()
            if victim_id is None:
                logger.warning("Cannot make room — all loaded models are pinned")
                break
            logger.info("Evicting model %s (LRU)", victim_id)
            await self._registry[victim_id].unload()
            del self._loaded_order[victim_id]

    def _find_lru_victim(self) -> str | None:
        """Find the least recently used non-pinned model."""
        for model_id in self._loaded_order:
            if model_id not in self._pinned:
                return model_id
        return None

    def _touch_lru(self, model_id: str) -> None:
        """Move model to end of LRU (most recently used). O(1) with OrderedDict."""
        if model_id in self._loaded_order:
            self._loaded_order.move_to_end(model_id)

    def list_models(self) -> list[dict[str, Any]]:
        """List all models with their status."""
        result = []
        for model_id, provider in self._registry.items():
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
