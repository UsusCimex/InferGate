# Архитектура

## Процессы

```
клиент ─▶ шлюз (app.main, порт 8000, без ML-библиотек) ─HTTP─▶ воркер модели (app.worker, порт 8001, один контейнер на модель)
```

- **Шлюз** — FastAPI: OpenAI-совместимые роутеры, планировщик запросов, менеджер моделей, кэш ответов, метрики. Образ лёгкий (`deploy/Dockerfile.gateway`), без torch.
- **Воркер** — тот же код, точка входа `app.worker`; читает один YAML (`WORKER_MODEL_CONFIG`), стартует без модели и загружает её по команде шлюза. У каждой модели свои зависимости (`deploy/workers/<id>/requirements.txt`), все воркеры собираются из одного `deploy/Dockerfile.worker`.
- Код Python запекается в образы: правка `app/` доходит до запущенных контейнеров только после пересборки. YAML моделей и папка весов монтируются, их правки видны сразу.

**Как шлюз находит воркер** (`ProviderManager`, по порядку): `worker_url` в YAML модели → переменная `WORKER_URL_<ID>` → `gpu.worker_url_template` → иначе модель создаётся локально в процессе шлюза (для разработки без Docker). Для удалённой модели шлюз создаёт `RemoteProvider` своей категории (`app/providers/remote.py`, `CATEGORY_REGISTRY`). Compose задаёт `WORKER_URL_*` всем моделям, поэтому каждая включённая модель регистрируется как удалённая; запрос к модели, чей контейнер не запущен, получает 503 `worker_not_ready`.

## Путь запроса

1. Роутер (`app/routers/`) определяет модель (из тела или `defaults`) и проверяет возможности до загрузки: картинки во входе, белый список голосов, «только клонирование».
2. Ключ кэша и поиск в кэше; при попадании — ответ сразу (`X-InferGate-Cache: HIT`), модель не загружается.
3. `ProviderManager.ensure_loaded` загружает модель, при необходимости вытеснив другие.
4. `GpuScheduler.submit` под `active_request`: слот модели, тайм-аут выполнения (`queue.timeout_seconds`; ответа воркера шлюз ждёт не меньше, а воркер, не ответивший вовремя, даёт 504).
5. Провайдер выполняет запрос (у удалённого — HTTP к воркеру), результат кладётся в кэш, добавляются заголовки `X-InferGate-*` и метрики Prometheus.

Клиент, оборвавший соединение, запрос не отменяет: uvicorn доводит обработчик до конца, результат попадает в кэш, и повтор того же запроса получает `HIT`. Так приложение, у которого истёк тайм-аут, при повторе с тем же seed забирает уже готовую картинку (проверено: клиент сдался через 1 с, повтор через 30 с — `HIT` за 26 мс). Без кэша (`strategy: never`, у текстовых моделей) результат оборванного запроса теряется.

Потоковый chat/completions загружает модель сразу и идёт мимо планировщика, кэша и метрик; шлюз передаёт поток воркера построчно, вместе с пустыми строками, которые разделяют события SSE.

## Воркер и его протокол

| Эндпоинт | Назначение |
|---|---|
| `GET /health` | всегда 200: `{"status": "ok"}` или `"loading"` |
| `GET /stats` | живая VRAM (NVML, иначе torch) и RAM |
| `POST /load` | 200, если модель готова; иначе 202 и загрузка в фоне |
| `GET /load/status` | `idle`, `loading`, `ready`, `failed`, `cancelling` |
| `POST /unload` | отменяет загрузку или выгружает модель |
| `POST /reload` | новый конфиг модели: без изменений, только метаданные или полная перезагрузка |
| `POST /generate`, `/synthesize`, `/voice-clone`, `/transcribe`, `/upscale`, `/embed`, `/embed-audio`, `/embed-image`, `/embed-video` | вывод; 503 `model_not_ready`, пока модель не загружена |

Запросы вывода воркер выполняет по одному (под общей блокировкой), поэтому `queue.max_concurrent` больше 1 в Docker не даёт параллельности. `ValueError` провайдера становится ответом 400 `invalid_request`, остальное — 500, которое шлюз передаёт клиенту.

**Загрузка со стороны шлюза.** `RemoteProvider.load`: `GET /health` (5 с; нет ответа — 503), `POST /load`, затем опрос `/load/status` (0.5, 1, 2, 2… с) до `ready`/`failed` или 1800 с. Если перезапущенный воркер отвечает `idle`, шлюз повторяет `POST /load`. JSON-запросы повторяются до 3 раз только при ошибках соединения; `X-Request-ID` передаётся воркеру.

## `ProviderManager` (`app/services/provider_manager.py`)

- **Загруженные модели** — `OrderedDict` в порядке использования; блокировка на модель и общая блокировка состояния.
- **Планировщик вытеснения**: модель помещается, если загружено меньше `gpu.max_loaded_models` GPU-моделей и (при ненулевом бюджете) сумма объявленных `vram_mb` с новой моделью не выходит за бюджет минус запас. Вытесняются давно не использованные GPU-модели с учётом `category_reservations`; закреплённые, модели с запросами в работе и модели на CPU (`device: cpu`, их `vram_mb` считается нулём) не трогаются. Если место занимают только модели с запросами в работе, загрузка ждёт конца этих запросов до `queue.timeout_seconds` загружаемой модели и строит план заново после каждого. Если плана нет и ждать нечего (мешают закреплённые модели) или ожидание вышло: при бюджете — 503 `insufficient_resources`, без бюджета — предупреждение и загрузка.
- **Монитор воркеров** раз в 10 с запрашивает `/health` (3 с). Первый ответ пишет в журнал `Worker reachable: <id>`. У загруженных моделей он проверяет `/load/status`: воркер, перезапустившийся без модели, освобождает слот, и следующий запрос загрузит модель заново. Загруженная модель считается потерянной после 3 пропусков подряд; неудачные проверки реже (10·2ⁿ⁻¹ с, до 300 с).
- **`reload_model`** вызывается `ConfigWatcher` при правке YAML: выключенная модель выгружается и снимается с регистрации, изменённая — перезагружается (удалённая и загруженная — через `/reload` воркера).

## `GpuScheduler` (`app/services/gpu_scheduler.py`)

Счётчик слотов на модель (`queue.max_concurrent`) и очередь ожидающих; общий предел `queue.max_size` (503 `queue_full`). Тайм-аут `queue.timeout_seconds` считается с момента получения слота (504). Очередь у каждой модели своя и фактически FIFO: приоритет из YAML одинаков у всех запросов к одной модели.

## Кэш (`app/services/cache_manager.py`, `cache_backends/`)

- Ключ — SHA-256 от модели и параметров запроса, которые прислал клиент (умолчания из YAML в ключ не входят). Для озвучки в ключ входят голос, скорость, формат, язык и seed; для клонирования — хэш референса и его текст; для распознавания — хэш аудио и параметры.
- Стратегии: `always`, `seed_only` (кэш только при `seed` в запросе — так повтор картинки с тем же seed отдаётся без генерации), `never` (LLM). Эмбеддинги не кэшируются.
- `X-InferGate-No-Cache: true` — пропустить кэш (`SKIP`).
- Локальный бэкенд: файлы `cache/<model>/<key[:2]>/<key>.<ext>` и метаданные в SQLite (WAL); запись через временный файл → коммит БД → переименование; вытеснение по `max_size_mb` модели и `max_total_size_gb`, TTL `cache.ttl_hours` модели. Есть бэкенд Redis (`cache.backend: redis`), но адрес задать нельзя — он всегда `redis://localhost:6379/0`.

## Сторожа

- **`ConfigWatcher`** — раз в 2 с проверяет `config/models/*.yaml` и вызывает `reload_model` и обновление пределов планировщика. Удалённый YAML только пишет предупреждение.
- **`MemoryWatchdog`** — см. [configuration.md](configuration.md#защита-памяти).

## Middleware

Все — чистый ASGI: `RequestIdMiddleware` (`X-Request-ID`), `PrometheusMiddleware`, `AccessLogMiddleware` (`INFERGATE_ACCESS_LOG_JSON=true` — журнал в JSON), а при включении — `ApiKeyMiddleware` и `RateLimitMiddleware`; CORS — стандартный Starlette.

Порядок снаружи внутрь: request id, журнал, метрики, CORS, лимит запросов, ключ API. Поэтому preflight `OPTIONS` проходит без ключа, а ответы 401 и 429 несут заголовки CORS и `X-Request-ID`. В Starlette внешним становится middleware, добавленный последним, — `create_app` добавляет их в обратном порядке.

## Структура кода

```
app/
├── main.py              шлюз: create_app, lifespan, обработчики ошибок, middleware
├── worker.py            воркер одной модели
├── dependencies.py      FastAPI Depends поверх app.state
├── config/              загрузка YAML (OmegaConf), схемы server и модели, перечисления
├── middleware/          auth, rate_limit, access_log
├── monitoring/          метрики Prometheus, X-Request-ID
├── routers/             chat, images, audio, embeddings, models, admin, cache, health
├── schemas/             pydantic-модели запросов и ответов (extra="forbid")
├── providers/
│   ├── base.py          базовые классы провайдеров
│   ├── registry.py      @register_provider
│   ├── remote.py        удалённые провайдеры шлюза; _remote_protocol.py — вызовы JSON и multipart
│   ├── image/           diffusers (+ _compel, _highres_fix, _lora, _schedulers, _textual_inversion), janus, meissonic
│   ├── text/            vLLM (+ _chat_images)
│   ├── tts/             kokoro, voxcpm2 (+ voxcpm2_voices/), qwen3_tts, xtts, fish_speech
│   ├── stt/             whisper
│   ├── upscale/         spandrel
│   └── embedding/       sentence-transformers, CLIP, SigLIP, CLAP, CLIP4Clip
├── services/            provider_manager, gpu_scheduler, cache_manager (+ cache_backends/), config_watcher, memory_watchdog
└── utils/uploads.py     ограничение размера загрузок
config/                  server.yaml, models/*.yaml, examples/
deploy/                  Dockerfile.gateway, Dockerfile.worker, docker-compose*.yml, docker-bake.hcl, workers/<id>/requirements.txt, monitoring/, Caddyfile.example, .env.example
scripts/                 diagnose/ (проверка одной модели), feature/ (сквозные проверки на живых контейнерах), benchmark.py, download_models.py
tests/                   pytest
```
