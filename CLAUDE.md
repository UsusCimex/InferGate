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

Client request → FastAPI router (`app/routers/`) → GPU Scheduler (priority queue) → Provider Manager (loads model if needed, LRU eviction) → Provider (`app/providers/`) → Response (with optional cache)

### Key Architectural Patterns

**Plugin-based providers**: Abstract bases in `app/providers/base.py` (`ImageProvider`, `TextProvider`, `TtsProvider`). New providers register via `@register_provider` decorator in `app/providers/registry.py`. Adding a model that uses an existing provider = just add a YAML file in `config/models/`.

**Module-level DI singletons**: Services initialized during FastAPI lifespan in `app/main.py`, stored in `app/dependencies.py`, accessed by routers via getter functions (`get_provider_manager()`, etc.).

**GPU memory management**: `ProviderManager` maintains an LRU cache of loaded models with a configurable max slot count. Models are auto-evicted when GPU is full. Pinned models (configured in `server.yaml`) are never evicted.

**Request scheduling**: `GpuScheduler` provides per-model concurrency semaphores with priority levels (high/medium/low) and configurable timeouts.

**Multi-strategy caching**: `CacheManager` uses disk storage + SQLite metadata. Strategies per model: `always`, `seed_only` (images with fixed seed), `never`. LRU eviction with TTL support.

### Core Services (app/services/)

- `provider_manager.py` — Model registry, loading/unloading, LRU GPU slot management
- `gpu_scheduler.py` — Request queue with priorities and concurrency limits
- `cache_manager.py` — Per-model caching with SQLite metadata index

### Configuration

- `config/server.yaml` — Global settings (auth, GPU slots, cache, CORS, rate limits, defaults)
- `config/models/*.yaml` — One file per model defining: hub_id, VRAM requirements, torch_dtype, default params, cache strategy, queue priority

### Custom Response Headers

- `X-InferGate-Cache`: HIT|MISS|DISABLED|SKIP
- `X-InferGate-Model`: actual model ID used
- `X-InferGate-Generation-Ms`: latency in milliseconds

## Tech Stack

- Python 3.11+, FastAPI, uvicorn
- vLLM (text), diffusers (image), kokoro/fish-speech (TTS)
- PyTorch with CUDA 12.6
- aiosqlite (cache metadata), pydantic (validation), ruff (linting)
- Docker with multi-layer build caching, uv package manager

## Testing

Tests use fake providers (`FakeImageProvider`, `FakeTextProvider`, `FakeTtsProvider`) defined in `tests/conftest.py`. The `services` fixture provides a fully initialized service layer; the `client` fixture provides an async FastAPI test client via httpx. Tests are async (pytest-asyncio).
