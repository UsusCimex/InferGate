# CLAUDE.md

InferGate is a self-hosted OpenAI-compatible gateway for local models: images (diffusers, Janus-Pro, Meissonic), text via vLLM (with image input for `capabilities.vision` models), TTS and voice cloning, speech-to-text (faster-whisper), upscaling (spandrel) and embeddings. A light FastAPI gateway proxies to one Docker worker container per model. 27 model configs live in `config/models/`; the main client is the PictoLex Android app (`../Comput`).

## Docs

User docs are in Russian under `docs/`: `api.md` (endpoints, fields, headers, errors), `models.md` (catalog, Compose profiles, providers, adding a model), `configuration.md` (`server.yaml`, model YAML, env vars, memory safety), `architecture.md` (gateway and worker, request flow, manager, scheduler, cache), `deployment.md` (Compose, images, TLS, monitoring), `testing.md`, `roadmap.md` (known issues and plans). Update the matching doc in the same commit as the behaviour change. Path-scoped rules live in `.claude/rules/`.

## Commands

```bash
pip install -e ".[dev]"                      # gateway + test deps (".[all]" adds torch/vLLM; Linux + CUDA only)
pytest                                       # no GPU, no network
pytest tests/test_chat.py
ruff check app/ tests/                       # what CI runs (Python 3.11 and 3.12)
uvicorn app.main:app --reload                # gateway; without WORKER_URL_* models load in-process
WORKER_MODEL_CONFIG=config/models/qwen3.5-4b.yaml uvicorn app.worker:app --port 8001

docker compose -f deploy/docker-compose.yml --profile text --profile tts up -d
docker compose -f deploy/docker-compose.yml --profile flux2-klein-4b up -d
docker buildx bake -f deploy/docker-bake.hcl worker-flux2-klein-4b
python scripts/download_models.py --models flux2-klein-4b   # prefetch weights on the host (HF_TOKEN in env)
```

## Architecture in brief

A request goes through the router (`app/routers/`), the capability checks, the cache lookup (a hit never loads the model), `ProviderManager.ensure_loaded` (LRU by declared `vram_mb`, budget, pinned models, per-model locks), `GpuScheduler.submit` inside `active_request`, the provider (`RemoteProvider` calls the worker over HTTP), then the cache put, `X-InferGate-*` headers and Prometheus metrics. Streaming chat bypasses scheduler, cache and metrics.

- Worker discovery order: YAML `worker_url`, env `WORKER_URL_<ID>`, `gpu.worker_url_template`, otherwise a local in-process provider.
- Workers start empty and load on demand (`POST /load` answers 202, then poll `/load/status`); the gateway's monitor probes `/health` every 10 s, drops a loaded model after 3 misses and frees the slot of a worker that restarted idle.
- Services are built in the FastAPI lifespan (`app/main.py`), kept on `app.state` and injected with `Depends()` from `app/dependencies.py`.
- Providers subclass the ABCs in `app/providers/base.py`, register with `@register_provider` and must sit in `app/providers/{image,text,tts,stt,upscale,embedding}/`. Every category also has a gateway-side remote class in `app/providers/remote.py` (`CATEGORY_REGISTRY`) and a worker endpoint in `app/worker.py`; keep the three in step.
- Cache: `CacheManager` facade over `LocalCacheBackend` (SQLite WAL + files) or `RedisCacheBackend`; strategies `always`, `seed_only`, `never` per model.
- Middleware is pure ASGI (request id, Prometheus, access log, optional API key and rate limit).

## Invariants that are easy to break

- **Code is baked into the Docker images**: a change in `app/` reaches running workers only after rebuilding their image. `config/models/*.yaml` and the weights folder are bind-mounted and apply immediately (YAML through `ConfigWatcher`).
- **`deploy/.env` holds `HF_TOKEN`**: never print it or commit it; show only filtered lines. A worker must be recreated to see a changed `.env`.
- **Adding a model** touches five places: the YAML, `deploy/workers/<id>/requirements.txt`, a `docker-bake.hcl` row, a `worker-<id>` service in `docker-compose.yml`, and `WORKER_URL_<ID>` on the gateway service. Env names are `<ID>_<FIELD>` with every non-alphanumeric character turned into `_`; service names turn dots into dashes.
- **`scripts/diagnose/*.sh` and several `scripts/feature/*.sh` rewrite `deploy/.env`** (profiles, quantization flags) without restoring it; check `.env` after running them.
- **PictoLex depends on the model ids, the `voice-clone` tag, `capabilities.voices` of `voxcpm2`, the TTS `seed` and the `X-InferGate-*` headers.** Renaming or dropping any of them breaks the app.

## Code style

- **Imports**: `from __future__ import annotations` first in every module, sorted by ruff (`I`), first-party is `app`.
- **Type hints**: PEP 604 and PEP 585 only (`dict[str, X]`, `list[X]`, `str | None`), never `Dict`, `List` or `Optional`.
- **Privacy**: `_foo` for module and class internals; double underscore only for deliberate name mangling.
- **Lint**: ruff with `E, W, F, I, UP, B, SIM, ASYNC, C4, T20, RUF`, line length 100 with `E501` ignored, no `print` in production code. `ruff check app/ tests/` must be clean before a commit.
- **Types**: `pyright` in `basic` mode; new code adds no `pyright` errors.
- **Async**: asyncio primitives (`asyncio.Lock`, `asyncio.Semaphore`) over threading; blocking filesystem or CPU work goes through `asyncio.to_thread`.

## Comments

Default to no comment.

- Docstrings: one line for classes, schemas and public functions (endpoints, provider ABCs, service entry points), saying what it does and, when the signature does not, what it returns or raises. None for private or trivial helpers and for modules.
- Inline `#` only for a non-obvious invariant, a workaround with its cause, a line whose removal silently breaks something, or `# noqa: XYZ` with a short reason.
- Forbidden: restating the code, section dividers, past bugs, PRs, incidents and dates, "chosen over X" rationale, licence or provenance prose, comments in model YAMLs and `pyproject.toml`.

If a comment does not fit one dense line, rename or restructure instead.

## Text style

Docs, comments, logs, error messages and commit messages: the necessary minimum, no intros, no restating the code. Keyboard characters only: a hyphen for em and en dashes, no arrows (not even `->` in prose), straight quotes, three dots for the ellipsis character, `x`, `~`, `>=` and `<=` for the math signs. Model prompts and parsed formats (SRT `-->`) stay as they are.

## Testing

Fake providers for every category live in `tests/conftest.py`; the `services` fixture builds manager, scheduler and cache on them, and `client` is an httpx `AsyncClient` over the app with `services` on `app.state` (no middleware). Tests are async (`asyncio_mode=auto`). Gateway and worker behaviour is tested in `tests/test_remote_e2e.py` against a fake worker over `ASGITransport`, and the real worker handlers in `tests/test_worker.py`. GPU behaviour is checked by `scripts/feature/*.sh` against live containers.

## Git

Commit straight to `main` (fast-forward), no PRs. Never add `Co-Authored-By` or other trailers.

Every commit title starts with a tag: `[*]` fix, logic or behaviour change; `[+]` addition (module, file, endpoint, dependency, test); `[-]` removal; `[r]` refactor without behaviour change. The title is one short sentence and ends with `:` only if a body follows; the body is a list of short `- ` bullets. One commit, one focused change.

    [*] Lazy-load models on first request:
    - drop eager provider.load() from worker lifespan
    - /health used only for probe

    [-] Legacy FishSpeech provider
