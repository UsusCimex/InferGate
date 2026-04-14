from __future__ import annotations

# Workaround: import setuptools before distutils to avoid vLLM assertion error
import setuptools  # noqa: F401

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse

from app.auth import ApiKeyMiddleware
from app.logging_middleware import AccessLogMiddleware
from app.rate_limit import RateLimitMiddleware
from app.config import load_model_configs, load_server_config
from app.routers import audio, cache, chat, health, images, models
from app.services.cache_manager import CacheManager
from app.services.gpu_scheduler import GpuScheduler, QueueFullError, RequestTimeoutError
from app.services.provider_manager import ModelNotFoundError, ProviderManager

logger = logging.getLogger("infergate")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize services on startup, cleanup on shutdown."""
    server_cfg = load_server_config()
    model_cfgs = load_model_configs()

    import warnings

    logging.basicConfig(
        level=getattr(logging, server_cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Suppress noisy warnings from third-party libraries
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Initialize services
    manager = ProviderManager(
        model_dir=server_cfg.models_dir,
        max_loaded=server_cfg.gpu.max_loaded_models,
        pinned=server_cfg.gpu.pinned_models,
    )
    manager.discover_models(model_cfgs)
    manager.validate_config()

    scheduler = GpuScheduler(max_queue_size=server_cfg.queue.max_size)
    for cfg in model_cfgs:
        if cfg.enabled:
            scheduler.register_model(cfg.id, cfg.queue.max_concurrent)

    cache_mgr = CacheManager(server_cfg.cache.model_dump())
    await cache_mgr.initialize()

    defaults = server_cfg.defaults.model_dump()

    app.state.provider_manager = manager
    app.state.gpu_scheduler = scheduler
    app.state.cache_manager = cache_mgr
    app.state.defaults = defaults
    app.state.start_time = time.time()

    # Preload default models so first requests are fast
    # Order: text (vLLM) first, then others — avoids distutils/setuptools conflicts
    preload_ids = list(dict.fromkeys(
        [defaults.get("text"), defaults.get("tts"), defaults.get("image")]
        + list(server_cfg.gpu.pinned_models)
    ))
    for model_id in preload_ids:
        if model_id is None:
            continue
        try:
            await manager.ensure_loaded(model_id)
            logger.info("Preloaded model: %s", model_id)
        except Exception as e:
            logger.warning("Failed to preload %s: %s", model_id, e)

    # Periodic TTL cleanup task
    cleanup_interval = server_cfg.cache.cleanup_interval_minutes * 60

    async def _cleanup_loop():
        while True:
            await asyncio.sleep(cleanup_interval)
            try:
                expired = await cache_mgr.invalidate_expired()
                if expired > 0:
                    logger.info("Cleaned up %d expired cache entries", expired)
            except Exception as e:
                logger.warning("Cache cleanup error: %s", e)

    cleanup_task = asyncio.create_task(_cleanup_loop())

    logger.info(
        "InferGate started — %d models registered, listening on %s:%d",
        len(model_cfgs),
        server_cfg.host,
        server_cfg.port,
    )

    yield

    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    await manager.shutdown()
    await cache_mgr.close()
    logger.info("InferGate stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="InferGate",
        description="Self-hosted OpenAI-compatible AI gateway",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Access logging (replaces noisy uvicorn default log)
    app.add_middleware(AccessLogMiddleware)

    # CORS — applied before auth so preflight requests work
    server_cfg = load_server_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_cfg.cors.allow_origins,
        allow_methods=server_cfg.cors.allow_methods,
        allow_headers=server_cfg.cors.allow_headers,
        allow_credentials=True,
    )

    # Rate limiting
    if server_cfg.rate_limit.enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=server_cfg.rate_limit.requests_per_minute,
        )

    # Auth
    if server_cfg.auth.enabled and server_cfg.auth.api_keys:
        app.add_middleware(ApiKeyMiddleware, api_keys=server_cfg.auth.api_keys)

    # Exception handlers
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

    # Routers
    app.include_router(chat.router)
    app.include_router(images.router)
    app.include_router(audio.router)
    app.include_router(models.router)
    app.include_router(cache.router)
    app.include_router(health.router)

    return app


app = create_app()
