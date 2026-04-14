from __future__ import annotations

import time
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import ModelConfig, ModelCacheConfig, ModelQueueConfig, ModelMetadata
from app.providers.base import ImageProvider, TextProvider, TtsProvider
from app.services.cache_manager import CacheManager
from app.services.gpu_scheduler import GpuScheduler
from app.services.provider_manager import ProviderManager


# --- Fake providers for testing ---

class FakeImageProvider(ImageProvider):
    async def load(self, model_dir: str) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def generate(self, prompt: str, **params: Any) -> bytes:
        # Return a minimal valid PNG (1x1 transparent pixel)
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )


class FakeTextProvider(TextProvider):
    async def load(self, model_dir: str) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def generate(self, messages: list[dict], **params: Any) -> dict:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! This is a test."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


class FakeTtsProvider(TtsProvider):
    async def load(self, model_dir: str) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def synthesize(self, text: str, **params: Any) -> bytes:
        # Return minimal WAV-like header
        return b"RIFF" + b"\x00" * 40


# --- Fixtures ---

def _make_model_config(
    model_id: str,
    category: str,
    provider_class: str,
    cache_strategy: str = "never",
) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        display_name=model_id,
        category=category,
        provider_class=provider_class,
        enabled=True,
        model={"hub_id": "test/test", "vram_mb": 1000},
        cache=ModelCacheConfig(enabled=cache_strategy != "never", strategy=cache_strategy, max_size_mb=100),
        queue=ModelQueueConfig(priority="medium", timeout_seconds=30, max_concurrent=2),
        metadata=ModelMetadata(),
    )


@pytest_asyncio.fixture
async def services(tmp_path):
    """Set up ProviderManager with fake providers, scheduler, and cache."""
    manager = ProviderManager(model_dir=str(tmp_path / "models"), max_loaded=3)

    # Manually register fake providers
    img_config = _make_model_config("test-image", "image", "FakeImageProvider", "seed_only")
    txt_config = _make_model_config("test-text", "text", "FakeTextProvider")
    tts_config = _make_model_config("test-tts", "tts", "FakeTtsProvider", "always")

    manager._registry["test-image"] = FakeImageProvider(img_config)
    manager._registry["test-text"] = FakeTextProvider(txt_config)
    manager._registry["test-tts"] = FakeTtsProvider(tts_config)

    scheduler = GpuScheduler(max_queue_size=10)
    scheduler.register_model("test-image", 2)
    scheduler.register_model("test-text", 2)
    scheduler.register_model("test-tts", 2)

    cache_mgr = CacheManager({
        "enabled": True,
        "directory": str(tmp_path / "cache"),
        "max_total_size_gb": 1,
        "eviction_policy": "lru",
    })
    await cache_mgr.initialize()

    defaults = {"image": "test-image", "text": "test-text", "tts": "test-tts"}

    yield {
        "manager": manager,
        "scheduler": scheduler,
        "cache": cache_mgr,
        "defaults": defaults,
    }

    await cache_mgr.close()


@pytest_asyncio.fixture
async def client(services):
    """Async test client for the FastAPI app (without lifespan, deps already set)."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from app.routers import chat, images, audio, models, cache, health
    from app.services.provider_manager import ModelNotFoundError
    from app.services.gpu_scheduler import RequestTimeoutError, QueueFullError

    app = FastAPI()

    # Set dependencies on app.state (used by Depends-based DI)
    app.state.provider_manager = services["manager"]
    app.state.gpu_scheduler = services["scheduler"]
    app.state.cache_manager = services["cache"]
    app.state.defaults = services["defaults"]
    app.state.start_time = time.time()

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request, exc):
        return JSONResponse(
            {"error": {"message": str(exc), "type": "not_found"}}, status_code=404
        )

    @app.exception_handler(RequestTimeoutError)
    async def timeout_handler(request, exc):
        return JSONResponse(
            {"error": {"message": str(exc), "type": "timeout"}}, status_code=504
        )

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(request, exc):
        return JSONResponse(
            {"error": {"message": str(exc), "type": "queue_full"}}, status_code=503
        )

    app.include_router(chat.router)
    app.include_router(images.router)
    app.include_router(audio.router)
    app.include_router(models.router)
    app.include_router(cache.router)
    app.include_router(health.router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
