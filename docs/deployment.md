# Развёртывание

## Docker Compose

```bash
cp deploy/.env.example deploy/.env      # HF_TOKEN, COMPOSE_PROFILES, тонкие настройки
docker compose -f deploy/docker-compose.yml up -d                     # профили из COMPOSE_PROFILES
docker compose -f deploy/docker-compose.yml --profile text --profile tts up -d
docker compose -f deploy/docker-compose.yml --profile flux2-klein-4b --profile qwen3.5-4b up -d
curl http://localhost:8000/health
```

- `gateway` стартует всегда (порт `${PORT:-8000}`), монтирует `config/` только для чтения и том `cache_data`; ему заданы `WORKER_URL_*` всех моделей.
- Каждый воркер монтирует свой YAML в `/app/config/model.yaml` и папку весов `${MODELS_DIR:-../models}` в `/app/models`. Общие блоки заданы YAML-якорями: окружение (`HF_TOKEN`, `PYTORCH_CUDA_ALLOC_CONF`), резервирование GPU, пределы памяти и `/dev/shm`, healthcheck через `python3` (curl в образах воркеров нет).
- Воркер стартует без модели, первый запрос или `POST /v1/models/{id}/load` её загружает. Веса скачиваются с Hugging Face в `MODELS_DIR` при первой загрузке; gated-модели (FLUX.1-dev, Llama) требуют `HF_TOKEN` и принятой лицензии.
- Веса можно скачать заранее в `models_dir` из `config/server.yaml`: `python scripts/download_models.py --models flux2-klein-4b kokoro-82m` (`--all` для всех включённых), `HF_TOKEN` в окружении.
- После правки `deploy/.env` воркер пересоздаётся: `docker compose -f deploy/docker-compose.yml up -d --force-recreate worker-<id>`.
- Docker Desktop: после перезагрузки хоста под сильной нагрузкой на диск чтение весов через bind mount может зависнуть, и загрузка модели не двигается. Помогает `docker restart` контейнера воркера.

Профили и модели: [models.md](models.md#профили-compose).

## Образы

- **`deploy/Dockerfile.gateway`**: `python:3.12-slim`, зависимости `requirements/base.txt`, код `app/` и `config/`, пользователь без root, `uvicorn app.main:app --port 8000`.
- **`deploy/Dockerfile.worker`**: один параметризованный Dockerfile для всех воркеров, пакеты ставятся через uv с кэшем BuildKit.

| ARG | Назначение | Пример |
|---|---|---|
| `BASE_IMAGE` | базовый образ | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` (по умолчанию для GPU), `vllm/vllm-openai:v0.19.0` (текст), `python:3.12-slim` (CPU-воркеры) |
| `APT_PACKAGES` | системные пакеты | `build-essential`, `git` |
| `WORKER_REQUIREMENTS` | requirements модели | `deploy/workers/flux2-klein-4b/requirements.txt` |
| `POST_INSTALL` | команда после pip | `python -m spacy download en_core_web_sm` (kokoro) |

Для RTX 30xx и 40xx: `GPU_BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`.

- **`deploy/docker-bake.hcl`**: матрица сборки, строка на модель, цели `worker-<id с дефисами>`, переменные `REGISTRY` и `TAG`.

```bash
docker buildx bake -f deploy/docker-bake.hcl                         # все воркеры параллельно
docker buildx bake -f deploy/docker-bake.hcl worker-flux2-klein-4b    # один
docker compose -f deploy/docker-compose.yml build worker-flux2-klein-4b
REGISTRY=myreg.io/infergate TAG=v1 docker buildx bake -f deploy/docker-bake.hcl --push
```

Код `app/` запекается в образ, после правки Python нужный образ пересобирается: правка кода занимает секунды, новая зависимость минуты, первая сборка модели 10-15 минут.

## HTTPS через Caddy

```bash
cp deploy/Caddyfile.example deploy/Caddyfile     # указать свой домен
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.tls.yml up -d
```

Caddy принимает 80 и 443 (HTTP/2, HTTP/3), выпускает сертификат Let's Encrypt, ограничивает тело запроса 200 МБ, сжимает ответы, передаёт `X-Request-ID` и закрывает снаружи `/metrics`. Порт 8000 шлюза наружу не публикуется (`ports: !reset []`, нужен Compose 2.20+). Для локальной сети без домена есть блок `tls internal` в `Caddyfile.example`; корневой сертификат Caddy (`/data/caddy/pki/authorities/local/root.crt`) добавляется в доверенные на клиентах. Ещё Caddy закрывает `/cache`, `/v1/admin/*` и загрузку и выгрузку моделей; с авторизацией шлюза этот блок можно убрать. Без авторизации стоит сузить `adapters.allowed_repos`.

## Мониторинг

```bash
docker compose -f deploy/docker-compose.yml -f deploy/monitoring/docker-compose.monitoring.yml --profile text up -d
```

- Prometheus: `http://localhost:9090`, раз в 15 с читает `gateway:8000/metrics/prometheus`.
- Grafana: `http://localhost:3000` (admin/admin). Источник данных и дашборд "InferGate - Gateway & Inference" подключаются автоматически (`deploy/monitoring/grafana/`): частота и задержка запросов, доля ошибок, p95 вывода по моделям, попадания в кэш, доступность воркеров и пул соединений к ним.

| Метрика | Тип | Метки |
|---|---|---|
| `infergate_requests_total` | counter | method, endpoint, status_code |
| `infergate_request_duration_seconds` | histogram | method, endpoint |
| `infergate_inference_duration_seconds` | histogram | model_id, category |
| `infergate_cache_hits_total`, `infergate_cache_misses_total` | counter | model_id |
| `infergate_models_loaded`, `infergate_gpu_vram_used_mb`, `infergate_queue_size` | gauge | |
| `infergate_worker_up` | gauge | model_id |
| `infergate_worker_health_check_duration_seconds` | histogram | model_id |
| `infergate_worker_disconnects_total` | counter | model_id, reason |
| `infergate_http_pool_connections` | gauge | model_id, state |
| `infergate_http_pool_waiting_requests` | gauge | model_id |
| `infergate_embedding_batch_inputs` | histogram | model_id |

Gauge-метрики шлюз обновляет при каждом чтении `/metrics/prometheus` и `/metrics`; VRAM берётся из `/stats` загруженных воркеров.

`infergate_worker_*` пишет монитор воркеров при каждой проверке `/health`: `up` 1, если воркер ответил 200; `reason` у выгруженных монитором моделей `unreachable` (3 пропуска подряд) или `restarted` (воркер перезапустился без модели). `infergate_http_pool_*` описывают пул соединений шлюза к каждому подключённому воркеру: соединения `active` и `idle` и запросы, которые ждут свободного соединения (предел `GATEWAY_REMOTE_MAX_CONNECTIONS`).

## Локально без Docker

```bash
pip install -e ".[dev]"                                        # шлюз и тесты
uvicorn app.main:app --reload                                  # шлюз; без WORKER_URL_* модели грузятся в его процессе
WORKER_MODEL_CONFIG=config/models/qwen3.5-4b.yaml uvicorn app.worker:app --port 8001   # отдельный воркер
```

Для локальной загрузки моделей нужны их зависимости (`gpu`, `tts`, `quant`, `embedding` в `pyproject.toml` или `deploy/workers/<id>/requirements.txt`); `.[all]` включает vLLM, который ставится только на Linux с CUDA.

## Требования к железу

| Конфигурация | GPU | RAM | Диск |
|---|---|---|---|
| Минимальная | 12 ГБ (RTX 3060, 5070) | 16-32 ГБ | 50 ГБ |
| Рекомендуемая | 24 ГБ (RTX 3090, 4090) | 32-64 ГБ | 100-200 ГБ |

На 12 ГБ крупные модели (FLUX.2 klein, FLUX.1-dev, Z-Image, Qwen-Image, Janus-Pro 7B) работают только в nf4, а языковая модель и модель картинок не помещаются одновременно, и шлюз меняет их в памяти.
