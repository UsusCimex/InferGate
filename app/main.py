from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import warnings
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import load_model_configs, load_server_config
from app.middleware import AccessLogMiddleware, ApiKeyMiddleware, RateLimitMiddleware
from app.monitoring import PrometheusMiddleware, RequestIdMiddleware
from app.routers import admin, audio, cache, chat, health, images, models
from app.services.cache_manager import CacheManager
from app.services.config_watcher import ConfigWatcher
from app.services.gpu_scheduler import GpuScheduler, QueueFullError, RequestTimeoutError
from app.services.memory_watchdog import MemoryWatchdog
from app.services.provider_manager import (
    InsufficientResourcesError,
    ModelNotFoundError,
    ProviderManager,
    WorkerNotReadyError,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize services on startup, cleanup on shutdown."""
    server_cfg = load_server_config()
    model_cfgs = load_model_configs()

    logging.basicConfig(
        level=getattr(logging, server_cfg.log_level.value.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    manager = ProviderManager(
        model_dir=server_cfg.models_dir,
        max_loaded=server_cfg.gpu.max_loaded_models,
        pinned=server_cfg.gpu.pinned_models,
        max_vram_budget_mb=server_cfg.gpu.max_vram_budget_mb,
        vram_headroom_mb=server_cfg.gpu.vram_headroom_mb,
        category_reservations=server_cfg.gpu.category_reservations,
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

    manager.start_worker_monitor()

    preload_ids = list(dict.fromkeys([
        defaults.get("text"),
        defaults.get("tts"),
        defaults.get("image"),
        *server_cfg.gpu.pinned_models,
    ]))
    for model_id in preload_ids:
        if model_id is None:
            continue
        try:
            if manager.get_config(model_id).worker_url:
                continue  # remote models are handled by worker monitor
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

    async def _on_config_reload(new_cfg):
        await manager.reload_model(new_cfg)
        scheduler.update_concurrency(new_cfg.id, new_cfg.queue.max_concurrent)

    config_watcher = ConfigWatcher("config/models", _on_config_reload)
    config_watcher.start()

    watchdog = MemoryWatchdog(
        manager=manager,
        interval_seconds=server_cfg.gpu.watchdog_interval_seconds,
        vram_threshold=server_cfg.gpu.watchdog_vram_threshold,
        ram_threshold=server_cfg.gpu.watchdog_ram_threshold,
    )
    watchdog.start()

    logger.info(
        "InferGate started — %d models registered, listening on %s:%d",
        len(model_cfgs),
        server_cfg.host,
        server_cfg.port,
    )

    yield

    await watchdog.stop()
    await config_watcher.stop()
    cleanup_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await cleanup_task
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

    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(RequestIdMiddleware)

    # CORS is added before auth so preflight requests pass without an API key.
    server_cfg = load_server_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_cfg.cors.allow_origins,
        allow_methods=server_cfg.cors.allow_methods,
        allow_headers=server_cfg.cors.allow_headers,
        allow_credentials=True,
    )

    if server_cfg.rate_limit.enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=server_cfg.rate_limit.requests_per_minute,
        )

    if server_cfg.auth.enabled and server_cfg.auth.api_keys:
        app.add_middleware(ApiKeyMiddleware, api_keys=server_cfg.auth.api_keys)

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request, exc):
        return JSONResponse(
            {"error": {"message": str(exc), "type": "not_found"}}, status_code=404
        )

    @app.exception_handler(WorkerNotReadyError)
    async def worker_not_ready_handler(request, exc):
        return JSONResponse(
            {"error": {"message": str(exc), "type": "worker_not_ready"}}, status_code=503
        )

    @app.exception_handler(InsufficientResourcesError)
    async def insufficient_resources_handler(request, exc):
        return JSONResponse(
            {"error": {"message": str(exc), "type": "insufficient_resources"}},
            status_code=503,
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

    @app.exception_handler(httpx.HTTPStatusError)
    async def upstream_error_handler(request, exc: httpx.HTTPStatusError):
        """Forward a worker's structured error response verbatim.

        When a remote worker returns 4xx/5xx with a JSON body (e.g. a
        ValueError surfaced by the worker as 400 `{"error": {...}}`),
        httpx raises HTTPStatusError inside the remote provider. Instead
        of letting that propagate to FastAPI's default 500 handler (which
        hides the root cause), we unwrap the body and mirror the worker's
        status so clients see the real reason.
        """
        resp = exc.response
        try:
            body = resp.json()
        except ValueError:
            body = {"error": {"message": resp.text or "Upstream error", "type": "upstream_error"}}
        return JSONResponse(body, status_code=resp.status_code)

    app.include_router(chat.router)
    app.include_router(images.router)
    app.include_router(audio.router)
    app.include_router(models.router)
    app.include_router(cache.router)
    app.include_router(health.router)
    app.include_router(admin.router)

    return app


app = create_app()
