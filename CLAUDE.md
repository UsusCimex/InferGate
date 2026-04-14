# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InferGate is a self-hosted OpenAI-compatible API gateway for local AI models. It serves image generation (FLUX, Stable Diffusion via diffusers), text generation (Qwen, Llama via vLLM), and text-to-speech (Kokoro, Fish Speech) through a unified REST API that any OpenAI SDK client can use by changing `base_url`.

## Commands

```bash
# Install (all dependencies including GPU + TTS + dev)
pip install -e ".[all]"

# Run dev server with hot-reload
uvicorn app.main:app --reload

# Run tests
pytest

# Run a single test file
pytest tests/test_chat.py

# Lint
ruff check app/

# Docker (GPU)
docker compose build
docker compose up -d

# Docker (CPU-only, TTS only)
docker compose -f docker-compose.cpu.yml up -d

# Download model weights
docker compose run --rm infergate python scripts/download_models.py --all
```

## Architecture

### Request Flow

Client request → FastAPI router (`app/routers/`) → GPU Scheduler (priority queue + asyncio.Lock) → Provider Manager (loads model if needed, LRU eviction with per-model locks) → Provider (`app/providers/`) → Response (with optional cache)

### Key Architectural Patterns

**Plugin-based providers**: Abstract bases in `app/providers/base.py` (`ImageProvider`, `TextProvider`, `TtsProvider`). New providers register via `@register_provider` decorator in `app/providers/registry.py`. Adding a model that uses an existing provider = just add a YAML file in `config/models/`.

**FastAPI Depends DI**: Services initialized during FastAPI lifespan in `app/main.py`, stored on `app.state`, accessed by routers via `Depends()` from `app/dependencies.py`. Supports `dependency_overrides` for testing.

**GPU memory management**: `ProviderManager` maintains an OrderedDict-based LRU cache of loaded models with O(1) touch/evict. Per-model locks allow concurrent model access while serializing per-model operations. Pinned models (configured in `server.yaml`) are never evicted.

**Request scheduling**: `GpuScheduler` provides per-model concurrency semaphores with asyncio.Lock-protected counters and configurable timeouts.

**Multi-strategy caching**: `CacheManager` uses disk storage + SQLite (WAL mode) metadata. Atomic writes (temp file → DB commit → rename). Strategies per model: `always`, `seed_only` (images with fixed seed), `never`. LRU eviction with TTL support. Accurate miss tracking via `cache_stats` table.

**Pure ASGI middleware**: All middleware (AccessLog, RateLimit, ApiKey) implemented as pure ASGI for ~20-40% lower overhead vs BaseHTTPMiddleware.

### Core Services (app/services/)

- `provider_manager.py` — Model registry, loading/unloading, OrderedDict LRU, per-model + state locks, shutdown timeouts
- `gpu_scheduler.py` — Request queue with asyncio.Lock-protected counters and concurrency limits
- `cache_manager.py` — Per-model caching with SQLite WAL metadata, atomic writes, miss tracking

### Configuration

- `config/server.yaml` — Global settings (auth, GPU slots, cache, CORS, rate limits, defaults)
- `config/models/*.yaml` — One file per model defining: hub_id, VRAM requirements, torch_dtype, trust_remote_code, default params, cache strategy, queue priority

### Custom Response Headers

- `X-InferGate-Cache`: HIT|MISS|DISABLED|SKIP
- `X-InferGate-Model`: actual model ID used
- `X-InferGate-Generation-Ms`: latency in milliseconds
- `X-InferGate-Queue-Position`: position in GPU scheduler queue

## Tech Stack

- Python 3.11+, FastAPI with Depends DI, uvicorn
- vLLM (text, with streaming SSE support), diffusers (image), kokoro/fish-speech (TTS)
- PyTorch with CUDA 12.6
- aiosqlite (cache metadata with WAL), pydantic (validation with Field constraints), ruff + pyright (linting + type checking)
- Docker with multi-layer build caching, uv package manager, non-root user

## Testing

Tests use fake providers (`FakeImageProvider`, `FakeTextProvider`, `FakeTtsProvider`) defined in `tests/conftest.py`. The `services` fixture provides a fully initialized service layer via `app.state`; the `client` fixture provides an async FastAPI test client via httpx. Tests are async (pytest-asyncio). Includes concurrency tests for scheduler, cache, and model loading.
