# Архитектура

## Процессы

Клиент обращается к шлюзу (`app.main`, порт 8000, без ML-библиотек), шлюз по HTTP к воркеру модели (`app.worker`, порт 8001, один контейнер на модель).

- **Шлюз**: FastAPI с OpenAI-совместимыми роутерами, планировщиком запросов, менеджером моделей, кэшем ответов и метриками. Образ без torch (`deploy/Dockerfile.gateway`).
- **Воркер**: тот же код с точкой входа `app.worker`. Читает один YAML (`WORKER_MODEL_CONFIG`), стартует без модели и загружает её по команде шлюза. Зависимости у каждой модели свои (`deploy/workers/<id>/requirements.txt`), образы собираются из общего `deploy/Dockerfile.worker`.
- Код Python запекается в образы: правка `app/` доходит до контейнеров только после пересборки. YAML моделей и папка весов монтируются, их правки видны сразу.

**Поиск воркера** (`ProviderManager`, по порядку): `worker_url` в YAML модели, переменная `WORKER_URL_<ID>`, `gpu.worker_url_template`, иначе модель создаётся в процессе шлюза (разработка без Docker). Для удалённой модели шлюз создаёт `RemoteProvider` её категории (`app/providers/remote.py`, `CATEGORY_REGISTRY`). Compose задаёт `WORKER_URL_*` всем моделям, поэтому каждая включённая модель удалённая; запрос к модели без запущенного контейнера получает 503 `worker_not_ready`.

## Путь запроса

1. Роутер (`app/routers/`) выбирает модель (из тела или `defaults`) и до загрузки проверяет возможности: картинки во входе, список голосов, "только клонирование".
2. Поиск в кэше. При попадании ответ сразу (`X-InferGate-Cache: HIT`), модель не загружается.
3. `ProviderManager.ensure_loaded` загружает модель, при необходимости вытеснив другие.
4. `GpuScheduler.submit` внутри `active_request`: слот модели и тайм-аут выполнения `queue.timeout_seconds`. Ответа воркера шлюз ждёт не меньше; не ответивший вовремя воркер даёт 504.
5. Провайдер выполняет запрос (удалённый по HTTP к воркеру), результат идёт в кэш, к ответу добавляются заголовки `X-InferGate-*`, пишутся метрики Prometheus.

Обрыв соединения клиентом запрос не отменяет: обработчик доходит до конца, результат попадает в кэш, и повтор того же запроса получает `HIT`. Так клиент после своего тайм-аута забирает готовую картинку повтором с тем же seed. Без кэша (`strategy: never`, текстовые модели) результат оборванного запроса теряется.

Потоковый chat/completions загружает модель сразу и идёт мимо планировщика, кэша и метрик. Поток воркера передаётся построчно вместе с пустыми строками, которые разделяют события SSE.

## Протокол воркера

| Эндпоинт | Назначение |
|---|---|
| `GET /health` | всегда 200: `{"status": "ok"}` или `"loading"` |
| `GET /stats` | живая VRAM (NVML, иначе torch) и RAM |
| `POST /load` | 200, если модель готова; иначе 202 и загрузка в фоне |
| `GET /load/status` | `idle`, `loading`, `ready`, `failed`, `cancelling` |
| `POST /unload` | отменяет загрузку или выгружает модель |
| `POST /reload` | новый конфиг: без изменений, только метаданные или полная перезагрузка; при полной старая модель выгружается до загрузки новой и возвращается, если новая не загрузилась |
| `POST /generate`, `/synthesize`, `/voice-clone`, `/transcribe`, `/upscale`, `/embed`, `/embed-audio`, `/embed-image`, `/embed-video` | вывод; 503 `model_not_ready`, пока модель не загружена |

Запросы вывода воркер выполняет по одному под общей блокировкой, потоковый ответ держит её до конца, поэтому `queue.max_concurrent` больше 1 в Docker параллельности не даёт, а `/reload` ждёт конца запросов. `ValueError` провайдера становится ответом 400 `invalid_request`, остальное ответом 500, которое шлюз передаёт клиенту.

**Загрузка со стороны шлюза** (`RemoteProvider.load`): `GET /health` (5 с, без ответа 503), `POST /load`, затем опрос `/load/status` (0.5, 1, 2, 2... с) до `ready` или `failed`, не дольше 1800 с. Если перезапущенный воркер отвечает `idle`, шлюз повторяет `POST /load`. JSON-запросы повторяются до 3 раз только при ошибках соединения; `X-Request-ID` передаётся воркеру.

## `ProviderManager` (`app/services/provider_manager.py`)

- **Загруженные модели**: `OrderedDict` в порядке использования; блокировка на модель и общая блокировка состояния.
- **Вытеснение**: модель помещается, если загружено меньше `gpu.max_loaded_models` GPU-моделей и (при ненулевом бюджете) сумма объявленных `vram_mb` вместе с ней не выходит за бюджет минус запас. Вытесняются давно не использованные GPU-модели с учётом `category_reservations`. Не трогаются закреплённые модели, модели с запросами в работе и модели на CPU (`device: cpu`, их `vram_mb` считается нулём). Если место занимают только модели с запросами в работе, загрузка ждёт конца этих запросов до `queue.timeout_seconds` загружаемой модели и строит план заново после каждого. Если плана нет и ждать нечего (мешают закреплённые) или ожидание вышло: с бюджетом 503 `insufficient_resources`, без бюджета предупреждение и загрузка.
- **Монитор воркеров**: раз в 10 с запрос `/health` (3 с). Первый ответ пишет в журнал `Worker reachable: <id>`. У загруженных моделей проверяется `/load/status`: воркер, перезапустившийся без модели, освобождает слот, и следующий запрос загрузит модель заново. Загруженная модель считается потерянной после 3 пропусков подряд; после неудач проверки идут реже (10 * 2^(n-1) с, до 300 с).
- **`reload_model`** вызывает `ConfigWatcher` при правке YAML: выключенная модель выгружается и снимается с регистрации, изменённая перезагружается (удалённая загруженная через `/reload` воркера).

## `GpuScheduler` (`app/services/gpu_scheduler.py`)

Слоты на модель (`queue.max_concurrent`) и очередь ожидающих; общий предел `queue.max_size` (503 `queue_full`). Тайм-аут `queue.timeout_seconds` отсчитывается от получения слота (504). Очередь у каждой модели своя: ждущие запросы выходят по приоритету (`X-InferGate-Priority`, без заголовка `queue.priority` модели), при равном приоритете по порядку прихода. Запросы `/v1/embeddings` к модели с `batching.enabled` встают в очередь пакетом, одним вызовом провайдера ([configuration.md](configuration.md#yaml-модели)).

## Кэш (`app/services/cache_manager.py`, `cache_backends/`)

- Ключ: SHA-256 от модели и параметров запроса поверх `default_params` модели, так что правка умолчаний в YAML даёт новые ключи. У озвучки в ключе голос, скорость, формат, язык и seed; у клонирования хэш референса и его текст; у распознавания хэш аудио и параметры.
- Стратегии: `always`, `seed_only` (кэш только при `seed` в запросе, повтор картинки с тем же seed отдаётся без генерации), `never` (LLM). Эмбеддинги не кэшируются.
- `X-InferGate-No-Cache: true` пропускает кэш (`SKIP`).
- Локальный бэкенд: файлы `cache/<model>/<key[:2]>/<key>.<ext>` и метаданные в SQLite (WAL). Запись: временный файл, коммит БД, переименование. Вытеснение по `max_size_mb` модели и `max_total_size_gb`, TTL `cache.ttl_hours` модели. Бэкенд Redis (`cache.backend: redis`, адрес `cache.redis_url`) годится для нескольких шлюзов с общим кэшем.

## Сторожа

- **`ConfigWatcher`** раз в 2 с проверяет `config/models/*.yaml` и вызывает `reload_model` и обновление пределов планировщика. Удаление YAML только пишет предупреждение.
- **`MemoryWatchdog`**: [configuration.md](configuration.md#защита-памяти).

## Middleware

Чистый ASGI: `RequestIdMiddleware` (`X-Request-ID`), `PrometheusMiddleware`, `AccessLogMiddleware`, по настройке `ApiKeyMiddleware` и `RateLimitMiddleware`; CORS стандартный из Starlette.

Порядок снаружи внутрь: request id, журнал, метрики, CORS, лимит запросов, ключ API. Поэтому preflight `OPTIONS` проходит без ключа, а ответы 401 и 429 несут заголовки CORS и `X-Request-ID`. В Starlette внешним становится последний добавленный middleware, поэтому `create_app` добавляет их в обратном порядке.

## Код

| Путь | Что внутри |
|---|---|
| `app/main.py` | шлюз: `create_app`, lifespan, обработчики ошибок, middleware |
| `app/worker.py` | воркер одной модели |
| `app/dependencies.py` | FastAPI `Depends` поверх `app.state` |
| `app/config/` | загрузка YAML (OmegaConf), схемы сервера и модели, перечисления |
| `app/middleware/` | auth, rate_limit, access_log |
| `app/monitoring/` | метрики Prometheus, `X-Request-ID` |
| `app/routers/` | chat, images, audio, embeddings, models, admin, cache, health |
| `app/schemas/` | pydantic-модели запросов и ответов (`extra="forbid"`) |
| `app/providers/` | `base.py` (базовые классы), `registry.py` (`@register_provider`), `remote.py` и `_remote_protocol.py` (удалённые провайдеры шлюза) |
| `app/providers/image/` | diffusers (`_compel`, `_highres_fix`, `_lora`, `_schedulers`, `_textual_inversion`), janus, meissonic |
| `app/providers/text/` | vLLM (`_chat_images`) |
| `app/providers/tts/` | kokoro, voxcpm2 (`voxcpm2_voices/`), qwen3_tts, xtts, fish_speech |
| `app/providers/stt/`, `upscale/`, `embedding/` | whisper; spandrel; sentence-transformers, CLIP, SigLIP, CLAP, CLIP4Clip |
| `app/services/` | provider_manager, gpu_scheduler, cache_manager (`cache_backends/`), config_watcher, memory_watchdog |
| `app/utils/uploads.py` | предел размера загрузок |
| `config/` | `server.yaml`, `models/*.yaml`, `examples/` |
| `deploy/` | Dockerfile шлюза и воркера, `docker-compose*.yml`, `docker-bake.hcl`, `workers/<id>/requirements.txt`, `monitoring/`, `Caddyfile.example`, `.env.example` |
| `scripts/` | `diagnose/` (одна модель), `feature/` (проверки на живых контейнерах), `benchmark.py`, `download_models.py` |
| `tests/` | pytest |
