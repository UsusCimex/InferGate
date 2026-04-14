from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any

from app.config import ModelConfig
from app.providers.base import BaseProvider
from app.providers.registry import get_provider_class

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
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

    def validate_config(self) -> None:
        """Validate that pinned models don't exceed max_loaded capacity."""
        gpu_pinned = [m for m in self._pinned if self._is_gpu_model(m)]
        if len(gpu_pinned) >= self._max_loaded:
            logger.warning(
                "Pinned GPU models (%d) >= max_loaded_models (%d). "
                "Non-pinned models cannot be loaded.",
                len(gpu_pinned),
                self._max_loaded,
            )

    def discover_models(self, configs: list[ModelConfig]) -> None:
        """Register providers from model configs."""
        for config in configs:
            if not config.enabled:
                logger.info("Skipping disabled model: %s", config.id)
                continue
            try:
                provider_cls = get_provider_class(config.provider_class)
                self._registry[config.id] = provider_cls(config)
                logger.info("Registered model: %s (%s)", config.id, config.provider_class)
            except ValueError as e:
                logger.warning("Failed to register %s: %s", config.id, e)

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

    def _get_model_lock(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._model_locks:
            self._model_locks[model_id] = asyncio.Lock()
        return self._model_locks[model_id]

    async def ensure_loaded(self, model_id: str) -> BaseProvider:
        """Load model if not loaded. LRU swap if no slots available."""
        # Fast path: already loaded — just touch LRU
        async with self._state_lock:
            provider = self.get(model_id)
            if provider.is_loaded():
                self._touch_lru(model_id)
                return provider

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
            result.append({
                "id": cfg.id,
                "object": "model",
                "created": 0,
                "owned_by": "infergate",
                "display_name": cfg.display_name,
                "category": cfg.category,
                "loaded": provider.is_loaded(),
                "enabled": cfg.enabled,
                "metadata": cfg.metadata.model_dump(),
            })
        return result

    def loaded_models(self) -> list[str]:
        """Return list of currently loaded model IDs."""
        return list(self._loaded_order)

    def resolve_default(self, category: str, defaults: dict[str, str]) -> str | None:
        """Resolve default model for a category."""
        return defaults.get(category)

    async def shutdown(self) -> None:
        """Unload all models on shutdown."""
        for model_id in list(self._loaded_order):
            try:
                await self._registry[model_id].unload()
            except Exception as e:
                logger.warning("Error unloading %s: %s", model_id, e)
        self._loaded_order.clear()
