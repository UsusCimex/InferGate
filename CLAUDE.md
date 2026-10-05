# CLAUDE.md

InferGate is a self-hosted OpenAI-compatible gateway for local models: images (diffusers, Janus-Pro, Meissonic), text via vLLM (with image input for `capabilities.vision` models), TTS and voice cloning, speech-to-text (faster-whisper), upscaling (spandrel) and embeddings. A light FastAPI gateway proxies to one Docker worker container per model. 27 model configs live in `config/models/`; the main client is the PictoLex Android app (`../Comput`).

## Docs

User docs are in Russian under `docs/`: `api.md` (endpoints, fields, headers, errors), `models.md` (catalog, Compose profiles, providers, adding a model), `configuration.md` (`server.yaml`, model YAML, env vars, memory safety), `architecture.md` (gateway/worker, request flow, manager, scheduler, cache), `deployment.md` (Compose, images, TLS, monitoring), `testing.md`, `roadmap.md` (known issues and plans). Update the matching doc in the same commit as the behaviour change. Path-scoped rules live in `.claude/rules/`.

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

Request flow: router (`app/routers/`) → capability checks → `ProviderManager.ensure_loaded` (LRU by declared `vram_mb`, budget, pinned models, per-model locks) → cache lookup → `GpuScheduler.submit` inside `active_request` → provider (`RemoteProvider` → worker HTTP) → cache put, `X-InferGate-*` headers, Prometheus metrics. Streaming chat bypasses scheduler, cache and metrics.

- Worker discovery: YAML `worker_url` → env `WORKER_URL_<ID>` → `gpu.worker_url_template` → otherwise a local in-process provider.
- Workers start empty and load on demand (`POST /load` → 202, poll `/load/status`); the gateway's monitor probes `/health` every 10 s, drops a loaded model after 3 misses and frees the slot of a worker that restarted idle.
- Services are built in the FastAPI lifespan (`app/main.py`), kept on `app.state` and injected with `Depends()` from `app/dependencies.py`.
- Providers subclass the ABCs in `app/providers/base.py`, register with `@register_provider` and must sit in `app/providers/{image,text,tts,stt,upscale,embedding}/`. Every category also has a gateway-side remote class in `app/providers/remote.py` (`CATEGORY_REGISTRY`) and a worker endpoint in `app/worker.py` — keep the three in step.
- Cache: `CacheManager` facade over `LocalCacheBackend` (SQLite WAL + files) or `RedisCacheBackend`; strategies `always` / `seed_only` / `never` per model.
- Middleware is pure ASGI (request id, Prometheus, access log, optional API key and rate limit).

## Invariants that are easy to break

- **Code is baked into the Docker images**: a change in `app/` reaches running workers only after rebuilding their image. `config/models/*.yaml` and the weights folder are bind-mounted and apply immediately (YAML through `ConfigWatcher`).
- **`deploy/.env` holds `HF_TOKEN`**: never print it or commit it; show only filtered lines. A worker must be recreated to see a changed `.env`.
- **Adding a model** touches five places: the YAML, `deploy/workers/<id>/requirements.txt`, a `docker-bake.hcl` row, a `worker-<id>` service in `docker-compose.yml`, and `WORKER_URL_<ID>` on the gateway service. Env names are `<ID>_<FIELD>` with every non-alphanumeric character turned into `_`; service names turn dots into dashes.
- **`scripts/diagnose/*.sh` and several `scripts/feature/*.sh` rewrite `deploy/.env`** (profiles, quantization flags) without restoring it — check `.env` after running them.
- **PictoLex depends on the model ids, the `voice-clone` tag, `capabilities.voices` of `voxcpm2`, the TTS `seed` and the `X-InferGate-*` headers.** Renaming or dropping any of them breaks the app.

## Code Style

Single canonical style — do not diverge.

- **Imports**: `from __future__ import annotations` first line in every module. Imports sorted by ruff (`I`), first-party = `app`.
- **Type hints**: PEP 604 / PEP 585 only — `dict[str, X]`, `list[X]`, `str | None`. Never `Dict`/`List`/`Optional` from `typing`.
- **Privacy**: single underscore `_foo` for module/class internals; double underscore only for deliberate name-mangling.
- **Line length**: 100 (`tool.ruff.line-length`). `E501` is ignored — format-first, wrap when it reads better.
- **Lint**: ruff with `E, W, F, I, UP, B, SIM, ASYNC, C4, T20, RUF`. No `print` in production code (`T20`). `ruff check app/ tests/` must be clean before commit.
- **Types**: `pyright` in `basic` mode (`tool.pyright`); new code should not introduce `pyright` errors.
- **Async**: prefer asyncio primitives (`asyncio.Lock`, `asyncio.Semaphore`) over threading. Wrap blocking filesystem/CPU work with `asyncio.to_thread`.

## Comment Policy

Comments exist to help a future reader understand a class, method, or inline gotcha without diving into the implementation — nothing more. Minimal information, informative, rare. These rules are strict.

### Docstrings

- **Classes / DTOs / Pydantic schemas**: one-line docstring stating the purpose. No multi-paragraph blurbs, no field-by-field lists, no "chosen over X because…" rationale. If a field is non-obvious, rename it.
- **Public methods and functions** (router endpoints, provider ABCs, public service entry points): one-line docstring stating what it does (and, when not obvious from the signature, what it returns or raises). Never restate parameter names.
- **Private / trivial helpers** (`_foo`, short getters, wrappers): no docstring. The name and type hints are the documentation.
- **Modules**: no module-level docstring. The filename + first class/function is enough.

### Inline `#` comments

Only legitimate reasons to keep one:

1. A **WHY** for a non-obvious invariant (e.g. ordering that exists for crash-safety, parameter-priority subtlety).
2. A **workaround** with its root cause (e.g. `# snapshot_download first — HF AutoModel skips speech_tokenizer/`).
3. A **footgun warning** where removing the line below would silently break something.
4. A `# noqa: XYZ` with a one-phrase reason.

Everything else is deleted. Specifically forbidden:
- Restating WHAT the next line does.
- Section dividers (`# ── Section ──`).
- References to past bugs, PRs, incidents, dates — that belongs in `git log`.
- "Chosen over X because Y" in docstrings (marketing, not contract).
- Release-date / license / provenance prose in docstrings.
- Comments on YAML configs and `pyproject.toml`. Those files are read by operators who understand their keys; a comment is noise.

If a would-be comment can't fit in one dense line that a reader genuinely needs, the code probably needs a better name or a more obvious structure instead.

## Testing

Fake providers for every category live in `tests/conftest.py`; the `services` fixture builds manager, scheduler and cache on them, and `client` is an httpx `AsyncClient` over the app with `services` on `app.state` (no middleware). Tests are async (`asyncio_mode=auto`). Gateway↔worker behaviour is tested in `tests/test_remote_e2e.py` against a fake worker over `ASGITransport`, and the real worker handlers in `tests/test_worker.py`. GPU behaviour is checked by `scripts/feature/*.sh` against live containers.

## Git Conventions

- Commit straight to `main` (fast-forward), no PRs. Do NOT add `Co-Authored-By` lines to commit messages.

### Commit Messages

Every commit starts with one tag:

- `[*]` — fix, logic change, feature behaviour
- `[+]` — addition (new module, file, endpoint, dependency, test)
- `[-]` — removal (deleted code, dropped feature)
- `[r]` — refactor (rename, move, restructure; no behaviour change)

Format:

- **Title**: `[tag] <one short sentence>`. End with `:` only if a body follows.
- **Body** (optional): dashed bullet list (`- ...`), one brief point per bullet.
- **Atomicity**: one commit = one focused change. Split, don't pad the message.

Example:

    [*] Lazy-load models on first request:
    - drop eager provider.load() from worker lifespan
    - /health used only for probe
    - remote and local share the same _make_room() path

    [-] Legacy FishSpeech provider

    [r] Move schedulers/compel/lora into submodules
