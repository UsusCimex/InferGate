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

# Docker — per-model isolated containers (gateway + workers with profiles)
docker compose -f deploy/docker-compose.yml --profile text --profile tts up -d

# Start specific models only
docker compose -f deploy/docker-compose.yml --profile qwen3.5-4b --profile kokoro-82m up -d

# Start all model categories
docker compose -f deploy/docker-compose.yml --profile text --profile image --profile tts up -d

# Gateway only (for external/remote workers)
docker compose -f deploy/docker-compose.yml up -d

# With monitoring
docker compose -f deploy/docker-compose.yml -f deploy/monitoring/docker-compose.monitoring.yml --profile text up -d

# Run a single worker locally for development
WORKER_MODEL_CONFIG=config/models/qwen3.5-4b.yaml uvicorn app.worker:app --port 8001

# Download model weights
docker compose -f deploy/docker-compose.yml run --rm worker-qwen3-5-4b python scripts/download_models.py --all
```

## Architecture

### Request Flow

Client request → FastAPI router (`app/routers/`) → GPU Scheduler (priority queue + asyncio.Lock) → Provider Manager (loads model if needed, LRU eviction with per-model locks) → Provider (`app/providers/`) → Response (with optional cache)

### Key Architectural Patterns

**Plugin-based providers**: Abstract bases in `app/providers/base.py` (`ImageProvider`, `TextProvider`, `TtsProvider`). New providers register via `@register_provider` decorator in `app/providers/registry.py`. Adding a model that uses an existing provider = just add a YAML file in `config/models/`.

**Per-model deployment**: Each model runs in its own isolated container with its own Dockerfile and dependencies (`deploy/workers/<model-id>/`). `ProviderManager` resolves worker URLs from environment variables (`WORKER_URL_<MODEL_ID>`) and creates `RemoteProvider` instances (from `app/providers/remote.py`) that proxy HTTP requests to standalone worker containers (`app/worker.py`). Gateway image is lightweight (~500MB, `deploy/Dockerfile.gateway`). Docker Compose profiles control which models to deploy (`--profile text`, `--profile kokoro-82m`, etc.).

**FastAPI Depends DI**: Services initialized during FastAPI lifespan in `app/main.py`, stored on `app.state`, accessed by routers via `Depends()` from `app/dependencies.py`. Supports `dependency_overrides` for testing.

**GPU memory management**: `ProviderManager` maintains an OrderedDict-based LRU cache of loaded models with O(1) touch/evict. Per-model locks allow concurrent model access while serializing per-model operations. Pinned models (configured in `server.yaml`) are never evicted.

**Request scheduling**: `GpuScheduler` provides per-model concurrency semaphores with asyncio.Lock-protected counters and configurable timeouts.

**Multi-strategy caching**: `CacheManager` uses disk storage + SQLite (WAL mode) metadata. Atomic writes (temp file → DB commit → rename). Strategies per model: `always`, `seed_only` (images with fixed seed), `never`. LRU eviction with TTL support. Accurate miss tracking via `cache_stats` table.

**Pure ASGI middleware**: All middleware (AccessLog, RateLimit, ApiKey) implemented as pure ASGI for ~20-40% lower overhead vs BaseHTTPMiddleware.

### Core Services (app/services/)

- `provider_manager.py` — Model registry, loading/unloading, OrderedDict LRU, per-model + state locks, shutdown timeouts
- `gpu_scheduler.py` — Request queue with asyncio.Lock-protected counters and concurrency limits
- `cache_manager.py` — Per-model caching with SQLite WAL metadata, atomic writes, miss tracking
- `config_watcher.py` — Polls `config/models/*.yaml` and calls back on change → `reload_model` + scheduler concurrency update
- `memory_watchdog.py` — Background task that polls per-worker VRAM + host RAM; emergency-evicts LRU when usage overshoots declared budgets

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

## Code Style

Single canonical style — do not diverge.

- **Imports**: `from __future__ import annotations` first line in every module. Imports sorted by ruff (`I`), first-party = `app`.
- **Type hints**: PEP 604 / PEP 585 only — `dict[str, X]`, `list[X]`, `str | None`. Never `Dict`/`List`/`Optional` from `typing`.
- **Privacy**: single underscore `_foo` for module/class internals; double underscore only for deliberate name-mangling.
- **Line length**: 100 (`tool.ruff.line-length`). `E501` is ignored — format-first, wrap when it reads better.
- **Lint**: ruff with `E, W, F, I, UP, B, SIM, ASYNC, C4, T20, RUF`. No `print` in production code (`T20`). `ruff check app/` must be clean before commit.
- **Types**: `pyright` in `basic` mode (`tool.pyright`); new code should not introduce `pyright` errors.
- **Async**: prefer asyncio primitives (`asyncio.Lock`, `asyncio.Semaphore`) over threading. Wrap blocking filesystem/CPU work with `asyncio.to_thread`.

## Comment Policy

Keep comments rare, short, and load-bearing. Three rules — everything else is noise.

1. **No comment by default.** Identifier names and types are already documentation.
2. **Comments explain WHY, never WHAT.** Only legitimate reasons to add one: a non-obvious invariant, a workaround with its cause, a parameter-priority subtlety, an ordering that exists for crash-safety, or behaviour that would surprise a reader. A comment that restates the line below it (`# Check TTL` above `if ttl_expires and ...`) must be deleted. No `# --- Section ---` dividers. No references to past fixes, incidents, or PRs — that belongs in `git log`.
3. **Docstrings only on public contracts** — classes, router endpoints, provider ABCs, public service methods. One-liner when it is enough; multi-line with Returns/Raises/Notes only when the contract genuinely needs it. No decorative docstrings on trivial helpers.

## Testing

Tests use fake providers (`FakeImageProvider`, `FakeTextProvider`, `FakeTtsProvider`) defined in `tests/conftest.py`. The `services` fixture provides a fully initialized service layer via `app.state`; the `client` fixture provides an async FastAPI test client via httpx. Tests are async (pytest-asyncio). Includes concurrency tests for scheduler, cache, and model loading.

## Git Conventions

- Do NOT add `Co-Authored-By` lines to commit messages. Keep commits clean — only the commit title and optional body describing what was done.
