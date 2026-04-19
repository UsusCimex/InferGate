# InferGate

Self-hosted OpenAI-совместимый AI-шлюз для локальных моделей. Единый gateway для генерации изображений, текста и озвучки.

Любое приложение, работающее с OpenAI API, может переключиться на InferGate сменой `base_url`.

---

## Оглавление

1. [Быстрый старт](#1-быстрый-старт)
2. [Использование](#2-использование)
3. [Поддерживаемые модели](#3-поддерживаемые-модели)
4. [API](#4-api)
5. [Кэширование](#5-кэширование)
6. [Добавление новой модели](#6-добавление-новой-модели)
7. [Конфигурация](#7-конфигурация)
8. [Архитектура](#8-архитектура)
9. [Distributed-режим](#9-distributed-режим)
10. [Мониторинг](#10-мониторинг)
11. [Docker-сборка](#11-docker-сборка)
12. [Тестирование](#12-тестирование)
13. [Системные требования](#13-системные-требования)
14. [TODO](#14-todo)
15. [Лицензия](#15-лицензия)

---

## 1. Быстрый старт

### Docker — изолированные контейнеры

Каждая модель работает в своём контейнере с собственными зависимостями. Docker Compose profiles позволяют выбрать, какие модели запускать.

```bash
git clone https://github.com/UsusCimex/infergate.git
cd infergate

# 1. Подготовить окружение (HF_TOKEN, профили, опциональные тюнинги)
cp deploy/.env.example deploy/.env
# отредактировать deploy/.env: HF_TOKEN=..., COMPOSE_PROFILES=qwen-image

# 2. Запустить (профиль читается из deploy/.env — COMPOSE_PROFILES)
docker compose -f deploy/docker-compose.yml up -d

# Или явно выбрать профиль:
docker compose -f deploy/docker-compose.yml --profile text --profile tts up -d
docker compose -f deploy/docker-compose.yml --profile qwen-image up -d

# Проверить
curl http://localhost:8000/health
```

Defaults в `config/models/*.yaml` заточены под 12GB GPU (nf4-квантизация крупных моделей, CPU offload). Для другого железа — задать env-переменные в `deploy/.env`: см. раздел [Конфигурация](#7-конфигурация).

#### Доступные profiles

| Profile | Модели |
|---------|--------|
| `text` | qwen3.5-4b (enabled) |
| `image` | sd35-medium (enabled) |
| `tts` | kokoro-82m (enabled) |
| `qwen3.5-4b`, `sd35-medium`, `kokoro-82m` | Индивидуальные |
| `qwen3.5-9b`, `qwen3-8b`, `llama3.1-8b`, `flux1-dev`, `flux1-schnell`, `flux2-klein-4b`, `openaudio-s1-mini` | Disabled по умолчанию, запуск через индивидуальный profile |

### Локальная разработка

```bash
pip install -e ".[all]"
uvicorn app.main:app --reload
```

После запуска:
- Сервер: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

---

## 2. Использование

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

# Текст
response = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Привет!"}]
)

# Текст (streaming)
stream = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Расскажи историю"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

# Изображение
response = client.images.generate(
    model="sd35-medium",
    prompt="Кот в космосе"
)

# Озвучка
response = client.audio.speech.create(
    model="kokoro-82m",
    input="Привет, мир!",
    voice="af_heart"
)
```

### curl

```bash
# Текст
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-4b", "messages": [{"role": "user", "content": "Привет!"}]}'

# Текст (streaming)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-4b", "messages": [{"role": "user", "content": "Привет!"}], "stream": true}'

# Изображение
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "sd35-medium", "prompt": "Кот в космосе"}'

# Озвучка
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro-82m", "input": "Привет, мир!"}' -o speech.mp3
```

---

## 3. Поддерживаемые модели

| Модель | Категория | Провайдер | VRAM | Лицензия |
|--------|-----------|-----------|------|----------|
| Stable Diffusion 3.5 Medium | Изображения | diffusers | 5 GB | Community |
| FLUX.1 Schnell | Изображения | diffusers | 8 GB | Apache 2.0 |
| FLUX.1 Dev | Изображения | diffusers | 12 GB | Non-Commercial |
| FLUX.2 Klein 4B | Изображения | diffusers | 6 GB | Apache 2.0 |
| Qwen 3.5 4B (AWQ) | Текст | vLLM | 4 GB | Apache 2.0 |
| Qwen 3.5 9B | Текст | vLLM | 8 GB | Apache 2.0 |
| Qwen 3 8B | Текст | vLLM | 7 GB | Apache 2.0 |
| Llama 3.1 8B | Текст | vLLM | 8 GB | Llama Community |
| Kokoro 82M | Озвучка | kokoro | CPU | MIT |
| OpenAudio S1 Mini | Озвучка | fish-speech | 4 GB | Apache 2.0 |

Любая diffusers/vLLM-совместимая модель добавляется одним YAML-файлом без написания кода.

---

## 4. API

Все эндпоинты совместимы с форматом OpenAI API.

### Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/v1/chat/completions` | Генерация текста (+ streaming) |
| `POST` | `/v1/images/generations` | Генерация изображений |
| `POST` | `/v1/audio/speech` | Синтез речи |
| `GET` | `/v1/models` | Список всех моделей |
| `POST` | `/v1/models/{id}/load` | Загрузить модель в GPU |
| `POST` | `/v1/models/{id}/unload` | Выгрузить модель |
| `GET` | `/cache/stats` | Статистика кэша |
| `DELETE` | `/cache` | Очистить весь кэш |
| `DELETE` | `/cache/{model_id}` | Очистить кэш модели |
| `DELETE` | `/cache/entry/{key}` | Удалить запись кэша |
| `GET` | `/health` | Проверка состояния |
| `GET` | `/metrics` | Метрики системы |

### Валидация параметров

| Параметр | Диапазон |
|----------|----------|
| `temperature` | 0.0 – 2.0 |
| `top_p` | 0.0 – 1.0 |
| `max_tokens` | 1 – 131072 |
| `n` (изображения) | 1 – 10 |
| `speed` (TTS) | 0.25 – 4.0 |

### Заголовки ответов

| Заголовок | Значения |
|-----------|----------|
| `X-InferGate-Cache` | `HIT`, `MISS`, `DISABLED`, `SKIP` |
| `X-InferGate-Model` | ID использованной модели |
| `X-InferGate-Queue-Position` | Позиция в очереди GPU |
| `X-InferGate-Generation-Ms` | Время генерации в мс |

### Заголовки запросов

| Заголовок | Описание |
|-----------|----------|
| `X-InferGate-No-Cache: true` | Пропустить кэш, сгенерировать заново |

---

## 5. Кэширование

Каждая модель определяет свою стратегию кэширования в YAML-конфиге:

| Стратегия | Описание | Применение |
|-----------|----------|------------|
| `always` | Кэшировать всегда | TTS |
| `seed_only` | Только если передан seed | Изображения |
| `never` | Не кэшировать | LLM |

Кэш хранится на диске с метаданными в SQLite (WAL-режим). Поддерживается LRU-вытеснение, TTL, атомарные записи (temp file → DB commit → rename).

### Инвалидация

| Действие | Как |
|----------|-----|
| Очистить весь кэш | `DELETE /cache` |
| Очистить кэш модели | `DELETE /cache/{model_id}` |
| Удалить одну запись | `DELETE /cache/entry/{key}` |
| Пропустить кэш (клиент) | Заголовок `X-InferGate-No-Cache: true` |
| Автоочистка по TTL | Автоматически (настраивается в `server.yaml`) |

---

## 6. Добавление новой модели

Четыре шага, ни одной строки Python/Dockerfile.

**1.** Создать `config/models/my-model.yaml` (env-переменные опциональны — default-ы работают сразу):

```yaml
id: my-model
display_name: "My Model"
category: image                                           # image | text | tts
provider_class: DiffusersImageProvider
enabled: ${oc.decode:${oc.env:MY_MODEL_ENABLED,true}}

model:
  hub_id: "org/model-name"
  vram_mb: ${oc.decode:${oc.env:MY_MODEL_VRAM_MB,8000}}
  torch_dtype: ${oc.env:MY_MODEL_TORCH_DTYPE,float16}
  quantization: ${oc.decode:${oc.env:MY_MODEL_QUANTIZATION,null}}
  cpu_offload: ${oc.decode:${oc.env:MY_MODEL_CPU_OFFLOAD,false}}
  default_params:
    num_inference_steps: ${oc.decode:${oc.env:MY_MODEL_STEPS,20}}

cache:
  enabled: true
  strategy: seed_only                                     # always | seed_only | never
  max_size_mb: 2048

queue:
  priority: ${oc.env:MY_MODEL_PRIORITY,low}
  timeout_seconds: ${oc.decode:${oc.env:MY_MODEL_TIMEOUT,120}}
  max_concurrent: ${oc.decode:${oc.env:MY_MODEL_MAX_CONCURRENT,1}}
```

**2.** Создать `deploy/workers/my-model/requirements.txt` с pip-зависимостями.

**3.** Добавить одну строку в матрицу `deploy/docker-bake.hcl`:

```hcl
{ id = "my-model", base = GPU_BASE_IMAGE, apt = "", post = "" },
```

**4.** Добавить ~15-строчный сервис в `deploy/docker-compose.yml` (скопировать любой соседний worker-stanza, поменять id, путь к config и requirements).

Перезапустить. Модель доступна через OpenAI API.

### Доступные провайдеры

| category | provider_class | Что поддерживает |
|----------|---------------|-----------------|
| `image` | `DiffusersImageProvider` | Любая diffusers-модель (FLUX, SD, PixArt и др.) |
| `text` | `VllmTextProvider` | Любая LLM через vLLM (Qwen, Llama, Mistral и др.) |
| `tts` | `KokoroTtsProvider` | Kokoro TTS |
| `tts` | `FishSpeechTtsProvider` | OpenAudio / Fish Speech |

> Если нужен провайдер для нового бэкенда, создайте класс в `app/providers/{категория}/`, наследуя `ImageProvider`, `TextProvider` или `TtsProvider`, и укажите его имя в `provider_class`.

### Build-args воркера

Все воркеры собираются из единого `deploy/Dockerfile.worker`, который принимает:

| ARG | Назначение | Пример |
|-----|-----------|--------|
| `BASE_IMAGE` | базовый образ | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, `vllm/vllm-openai:v0.19.0`, `python:3.12-slim` |
| `APT_PACKAGES` | доп. apt-пакеты | `build-essential` (для bitsandbytes/triton), `git`, `gcc` |
| `WORKER_REQUIREMENTS` | путь к requirements.txt | `deploy/workers/my-model/requirements.txt` |
| `POST_INSTALL` | shell-команда после pip install | `python -m spacy download en_core_web_sm` |

Не надо копировать Dockerfile на каждую новую модель.

---

## 7. Конфигурация

### `deploy/.env` — переносимые тюнинги железа

Все hardware-чувствительные поля в `config/models/*.yaml` параметризованы через OmegaConf: `${oc.env:VAR,default}` (строки) и `${oc.decode:${oc.env:VAR,default}}` (bool/int/null). Это значит, что для переноса на другое железо **ни один YAML в репозитории редактировать не нужно** — меняются только env-переменные в `deploy/.env`.

Базовый сценарий:

```bash
cp deploy/.env.example deploy/.env
# отредактировать deploy/.env
docker compose -f deploy/docker-compose.yml up -d
```

Примеры переключения режимов:

| Сценарий | Что поменять в `deploy/.env` |
|---|---|
| **Laptop 12GB** (default) | ничего — defaults laptop-safe |
| **Server 24GB**, qwen-image без квантизации | `QWEN_IMAGE_QUANTIZATION=null`, `QWEN_IMAGE_CPU_OFFLOAD=false`, `QWEN_IMAGE_MAX_CONCURRENT=2` |
| **Server 48GB**, FLUX.1 dev на полной скорости | `FLUX1_DEV_CPU_OFFLOAD=false`, `FLUX1_DEV_STEPS=50` |
| **Shared GPU** | `QWEN3_5_4B_GPU_MEM_UTIL=0.50` |
| **Большой контекст** | `QWEN3_5_4B_CONTEXT_LENGTH=32768` |
| **RTX 30xx/40xx** | `GPU_BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime` |

Полный каталог переменных с комментариями — в [`deploy/.env.example`](deploy/.env.example).

Соглашение имени: `<MODEL_ID_UPPER>_<FIELD>` (дефисы и точки в ID → `_`). Например `qwen-image` → `QWEN_IMAGE_*`, `qwen3.5-4b` → `QWEN3_5_4B_*`.

### Host-specific overrides — compose override-файлы

Для вещей, которые не выразить через env (GPU count, volumes, deploy-секции), — override-файл поверх базового compose:

```bash
docker compose -f deploy/docker-compose.yml \
               -f deploy/docker-compose.server.yml \
               up -d
```

Образец — [`deploy/docker-compose.server.example.yml`](deploy/docker-compose.server.example.yml).

### `config/server.yaml`

```yaml
host: "0.0.0.0"
port: 8000
log_level: info

auth:
  enabled: false
  api_keys: []                     # ["key1", "key2"]

gpu:
  max_loaded_models: 3             # Макс моделей в GPU одновременно
  pinned_models: []                # Модели, которые не выгружаются

cache:
  enabled: true
  directory: "./cache"
  max_total_size_gb: 10
  eviction_policy: lru
  cleanup_interval_minutes: 30

cors:
  allow_origins: ["*"]
  allow_methods: ["*"]
  allow_headers: ["*"]

defaults:                          # Модели по умолчанию (если model не указан)
  image: sd35-medium
  text: qwen3.5-4b
  tts: kokoro-82m

rate_limit:
  enabled: false
  requests_per_minute: 60
```

---

## 8. Архитектура

```
Client → FastAPI (pure ASGI middleware: auth, rate-limit, access-log)
       → Router → GPU Scheduler (asyncio.Lock + semaphores)
       → Provider Manager (OrderedDict LRU, per-model locks)
       → Provider (local or remote) → Response (+ cache)
```

### Структура проекта

```
infergate/
├── app/
│   ├── main.py                     # FastAPI, lifespan, exception handlers
│   ├── worker.py                   # Standalone worker для distributed-режима
│   ├── config.py                   # Загрузка YAML-конфигов
│   ├── auth.py                     # API Key middleware (pure ASGI)
│   ├── rate_limit.py               # Rate limiter (pure ASGI)
│   ├── logging_middleware.py       # Access log (pure ASGI)
│   ├── dependencies.py             # FastAPI Depends + app.state
│   ├── routers/                    # API эндпоинты
│   │   ├── chat.py                 # /v1/chat/completions (+ streaming)
│   │   ├── images.py               # /v1/images/generations
│   │   ├── audio.py                # /v1/audio/speech
│   │   ├── models.py               # /v1/models
│   │   ├── cache.py                # /cache/*
│   │   └── health.py               # /health, /metrics
│   ├── schemas/                    # Pydantic-модели с Field-валидацией
│   ├── providers/
│   │   ├── base.py                 # ABC: ImageProvider, TextProvider, TtsProvider
│   │   ├── registry.py             # @register_provider + авто-обнаружение
│   │   ├── remote.py               # RemoteProvider для distributed-режима
│   │   ├── image/
│   │   │   └── diffusers_provider.py
│   │   ├── text/
│   │   │   └── vllm_provider.py    # + streaming + tokenizer chat templates
│   │   └── tts/
│   │       ├── kokoro.py
│   │       └── fish_speech.py
│   └── services/
│       ├── provider_manager.py     # LRU (OrderedDict), per-model locks, shutdown timeout
│       ├── gpu_scheduler.py        # asyncio.Lock-protected counters + semaphores
│       └── cache_manager.py        # SQLite WAL, atomic writes, miss tracking
├── config/
│   ├── server.yaml
│   └── models/                     # 1 YAML = 1 модель
├── deploy/
│   ├── Dockerfile.gateway          #   Лёгкий gateway (~500MB)
│   ├── Dockerfile.worker           #   Единый параметризованный Dockerfile для всех воркеров
│   ├── docker-compose.yml          #   Compose с profiles (gateway + workers, YAML-якоря)
│   ├── docker-bake.hcl             #   Матрица сборки (1 строка = 1 модель)
│   ├── .env.example                #   Каталог env-тюнингов
│   ├── docker-compose.server.example.yml  # Образец override для сервера
│   ├── workers/                    #   Только requirements.txt per-model
│   │   ├── qwen3.5-4b/requirements.txt
│   │   ├── sd35-medium/requirements.txt
│   │   ├── kokoro-82m/requirements.txt
│   │   └── ...                     #   (11 моделей)
│   └── monitoring/                 #   Prometheus + Grafana stack
│       ├── prometheus.yml
│       └── docker-compose.monitoring.yml
├── tests/                          # 98 тестов, 82% покрытия
└── pyproject.toml
```

---

## 9. Per-model архитектура

Каждая модель работает в изолированном контейнере с собственным Dockerfile и зависимостями. Gateway — лёгкий образ (~500MB) без ML-библиотек.

```
Client → Gateway (500MB, без GPU)
           ├→ worker-qwen3-5-4b     (vLLM, GPU)
           ├→ worker-sd35-medium    (diffusers, GPU)
           ├→ worker-kokoro-82m     (kokoro, CPU)
           └→ ...
```

### Как это работает

Gateway определяет worker URL из переменных окружения (задаются в `docker-compose.yml`):

```
WORKER_URL_QWEN3_5_4B=http://worker-qwen3-5-4b:8001
WORKER_URL_SD35_MEDIUM=http://worker-sd35-medium:8001
```

Формула: `WORKER_URL_` + model ID в верхнем регистре, `-` и `.` заменяются на `_`.

При наличии env var `ProviderManager` создаёт `RemoteProvider`, который проксирует HTTP к воркеру. Без env var — модель загружается локально (для разработки без Docker).

### Запуск worker вручную

```bash
WORKER_MODEL_CONFIG=config/models/qwen3.5-4b.yaml \
uvicorn app.worker:app --host 0.0.0.0 --port 8001
```

### Структура воркера

```
deploy/
├── Dockerfile.worker            # Один параметризованный Dockerfile на все воркеры
├── docker-bake.hcl              # Матрица: 1 строка — 1 модель
└── workers/
    └── qwen3.5-4b/
        └── requirements.txt     # Только pip-зависимости этой модели
```

---

## 10. Мониторинг

### Встроенные метрики

| Эндпоинт | Формат | Описание |
|----------|--------|----------|
| `GET /metrics` | JSON | Snapshot: очередь, VRAM, loaded models, cache hit rate, uptime |
| `GET /metrics/prometheus` | Prometheus text | Time-series для scraping |
| `GET /health` | JSON | Проверка состояния (503 если БД недоступна) |

Каждый ответ содержит заголовок `X-Request-ID` для сквозного трейсинга.

### Prometheus-метрики

| Метрика | Тип | Labels | Описание |
|---------|-----|--------|----------|
| `infergate_requests_total` | Counter | method, endpoint, status_code | Общее количество HTTP-запросов |
| `infergate_request_duration_seconds` | Histogram | method, endpoint | Длительность запросов |
| `infergate_inference_duration_seconds` | Histogram | model_id, category | Время инференса моделей |
| `infergate_cache_hits_total` | Counter | model_id | Попадания в кэш |
| `infergate_cache_misses_total` | Counter | model_id | Промахи кэша |
| `infergate_models_loaded` | Gauge | — | Количество загруженных моделей |
| `infergate_gpu_vram_used_mb` | Gauge | — | Использование GPU VRAM |
| `infergate_queue_size` | Gauge | — | Размер очереди GPU |

### Запуск с Prometheus + Grafana

```bash
docker compose -f deploy/docker-compose.yml -f deploy/monitoring/docker-compose.monitoring.yml --profile text up -d
```

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

`prometheus-client` — optional dependency. Без неё метрики деградируют до JSON `/metrics`, Prometheus endpoint возвращает 501.

```bash
pip install infergate[monitoring]
```

---

## 11. Docker-сборка

Сборка построена на принципе **одно описание — много вариантов**:

- **Единый `deploy/Dockerfile.worker`** — параметризован build-args (`BASE_IMAGE`, `APT_PACKAGES`, `WORKER_REQUIREMENTS`, `POST_INSTALL`). Все 11 воркеров собираются из него — нет копипаста.
- **`deploy/docker-bake.hcl`** — матрица сборки (HCL), 1 модель = 1 строка. `docker buildx bake` собирает все параллельно, с общим кэшем слоёв и поддержкой registry push.
- **`deploy/docker-compose.yml`** с YAML-якорями для общих блоков (environment, healthcheck, GPU reservations).
- **uv** вместо pip — установка в 10–100x быстрее.
- **BuildKit cache mounts** — пакеты переиспользуются между сборками.

### Команды сборки

```bash
# Собрать все воркеры параллельно через bake
docker buildx bake -f deploy/docker-bake.hcl

# Собрать одну модель
docker buildx bake -f deploy/docker-bake.hcl worker-qwen-image

# Push в registry
REGISTRY=myreg.io/infergate TAG=v1 \
  docker buildx bake -f deploy/docker-bake.hcl --push

# Либо по-compose-овски (одна модель)
docker compose -f deploy/docker-compose.yml build worker-qwen-image
```

### Скорость пересборки

| Сценарий | Время |
|----------|-------|
| Первая сборка (одна модель) | ~10–15 мин |
| Изменение кода (app/) | ~10 сек |
| Новая зависимость модели | ~5–10 мин (только эта модель) |
| Добавление новой модели | Не затрагивает существующие |

---

## 12. Тестирование

```bash
pip install -e ".[dev]"
pytest

# С покрытием
pip install pytest-cov
pytest --cov=app --cov-branch
```

**98 тестов**, **82% покрытия** (ветвевое). Покрытие исключает GPU-провайдеры, которые требуют физическое GPU для интеграционного тестирования.

Тесты покрывают: роутеры, middleware (auth, rate-limit, access-log), мониторинг (Prometheus, request ID), cache manager, GPU scheduler, provider manager, конфигурацию, валидацию, worker, конкурентность, edge cases.

---

## 13. Системные требования

| Конфигурация | GPU | RAM | Диск |
|-------------|-----|-----|------|
| Минимальная | RTX 3060 12 GB | 16 GB | 50 GB |
| Рекомендуемая | RTX 3090 24 GB | 32 GB | 100 GB |
| Оптимальная | RTX 4090 24 GB | 64 GB | 200 GB |

---

## 14. TODO

### Расширения API и семплинга (итерация 1 — быстрые победы)

- [x] **Per-request tunables в API** — `negative_prompt`, `num_inference_steps`, `guidance_scale` прокинуты через `ImageGenerationRequest`
- [x] **Scheduler swap per-request** — таблица имя→класс (Euler, DPM++, DDIM, LMS, Heun, UniPC и др.), поле `scheduler` в запросе, своп через `Cls.from_config(pipe.scheduler.config)` внутри GPU-executor потока
- [ ] **A1111-style token weighting** — синтаксис `(word:1.2)` через `compel` библиотеку для prompt-embedding с весами

### LoRA и тонкая настройка (итерация 2 — средняя сложность)

- [ ] **LoRA hot-load** — `{"loras": [{"id": "user/style-anime", "weight": 0.8}]}` в запросе, per-request загрузка через `pipe.load_lora_weights + pipe.set_adapters`, LRU-кэш скачанных LoRA, graceful unload
- [ ] **Textual Inversion** — `pipe.load_textual_inversion` для новых token embeddings
- [ ] **LyCORIS** — расширенные LoRA через `peft` библиотеку

### Мульти-стадийный пайплайн (итерация 3 — большая работа)

- [ ] **SDXL Refiner** — base model → refiner model ensemble, передача latents между ними (`SDXLRefinerImageProvider`)
- [ ] **HighresFix** — двухэтапная генерация: low-res → upscale → img2img refine
- [ ] **Upscaler provider** — Real-ESRGAN / SwinIR как отдельная категория, post-processing шаг

### Инфраструктура

- [ ] **End-to-end error forwarding** — ValueError в воркер-провайдере сейчас возвращается как generic HTTP 500 "Internal Server Error". Надо: воркер сериализует ValueError в 400/422 + JSON body, gateway пробрасывает status + body клиенту через `httpx.HTTPStatusError` → `JSONResponse(e.response.json(), status_code=e.response.status_code)`
- [ ] **Web UI** — панель администрирования
- [ ] **Voice cloning** — клонирование голоса через XTTS-v2 / OpenAudio S1
- [ ] **Speech-to-Text** — эндпоинт `/v1/audio/transcriptions`
- [ ] **Multi-GPU** — распределение моделей по нескольким GPU (CUDA device_ids)
- [ ] **Hot-reload конфигов** — добавление моделей без перезапуска сервера
- [ ] **Kubernetes Helm chart** — для multi-node distributed-режима
- [ ] **Grafana дашборд** — готовый JSON-дашборд для импорта

---

## 15. Лицензия

MIT
