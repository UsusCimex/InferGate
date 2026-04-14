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
10. [Docker-сборка](#10-docker-сборка)
11. [Тестирование](#11-тестирование)
12. [Системные требования](#12-системные-требования)
13. [TODO](#13-todo)
14. [Лицензия](#14-лицензия)

---

## 1. Быстрый старт

### Docker (GPU) — монолит

```bash
git clone https://github.com/UsusCimex/infergate.git
cd infergate

# Собрать образ
docker compose build

# Скачать веса моделей
docker compose run --rm infergate python scripts/download_models.py --all

# Запустить
docker compose up -d

# Проверить
curl http://localhost:8000/health
```

### Docker (Distributed) — изолированные контейнеры

Каждая модель в своём контейнере с собственными зависимостями. Нет конфликтов библиотек.

```bash
docker compose -f docker-compose.distributed.yml up -d
```

### Docker (только CPU)

```bash
docker compose -f docker-compose.cpu.yml up -d
```

Лёгкий образ без GPU-зависимостей. Доступны только TTS-модели (Kokoro).

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

Создать файл `config/models/my-model.yaml`:

```yaml
id: my-model
display_name: "My Model"
category: image                    # image | text | tts
provider_class: DiffusersImageProvider
enabled: true

model:
  hub_id: "org/model-name"
  vram_mb: 8000
  torch_dtype: float16
  trust_remote_code: false         # явно включать только для доверенных моделей
  default_params:
    num_inference_steps: 20

cache:
  enabled: true
  strategy: seed_only              # always | seed_only | never
  max_size_mb: 2048

queue:
  priority: low                    # high | medium | low
  timeout_seconds: 120
  max_concurrent: 1
```

Перезапустить сервер — модель доступна. Писать код не нужно.

### Доступные провайдеры

| category | provider_class | Что поддерживает |
|----------|---------------|-----------------|
| `image` | `DiffusersImageProvider` | Любая diffusers-модель (FLUX, SD, PixArt и др.) |
| `text` | `VllmTextProvider` | Любая LLM через vLLM (Qwen, Llama, Mistral и др.) |
| `tts` | `KokoroTtsProvider` | Kokoro TTS |
| `tts` | `FishSpeechTtsProvider` | OpenAudio / Fish Speech |

> Если нужен провайдер для нового бэкенда, создайте класс в `app/providers/{категория}/`, наследуя `ImageProvider`, `TextProvider` или `TtsProvider`, и укажите его имя в `provider_class`.

### Зависимости

Зависимости разделены на слои для оптимизации Docker-сборки:

| Файл | Содержимое |
|------|-----------|
| `requirements/ml.txt` | ML-библиотеки (vllm, diffusers, transformers) |
| `requirements/tts.txt` | TTS-библиотеки (kokoro, soundfile) |
| `requirements/base.txt` | Серверные зависимости (fastapi, pydantic) |

---

## 7. Конфигурация

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
├── tests/                          # 91 тест, 82% покрытия
├── Dockerfile                      # Монолитный GPU-образ
├── Dockerfile.gateway              # Лёгкий gateway (~500MB)
├── Dockerfile.worker               # Worker с параметризуемыми зависимостями
├── docker-compose.yml              # Монолит
├── docker-compose.cpu.yml          # CPU-only
├── docker-compose.distributed.yml  # Distributed (gateway + workers)
└── pyproject.toml
```

---

## 9. Distributed-режим

Каждая модель работает в изолированном контейнере с собственным набором зависимостей. Решает проблему конфликтов версий библиотек.

```
Client → Gateway (500MB, без GPU)
           ├→ worker-text   (vLLM + transformers, GPU)
           ├→ worker-image  (diffusers + torch, GPU)
           └→ worker-tts    (kokoro, CPU, python:3.12-slim)
```

### Запуск

```bash
docker compose -f docker-compose.distributed.yml up -d
```

### Как подключить модель к worker

В YAML-конфиге модели добавьте `worker_url`:

```yaml
# config/models/qwen3.5-4b.yaml (на стороне gateway)
id: qwen3.5-4b
worker_url: "http://worker-text:8001"  # gateway проксирует к этому worker
# ... остальные поля ...
```

Без `worker_url` — модель загружается локально (как в монолитном режиме).

### Запуск worker вручную

```bash
WORKER_MODEL_CONFIG=config/models/qwen3.5-4b.yaml \
uvicorn app.worker:app --host 0.0.0.0 --port 8001
```

### Преимущества

| | Монолит | Distributed |
|---|---|---|
| Изоляция зависимостей | Нет | Полная |
| Размер gateway-образа | ~15 GB | ~500 MB |
| Конфликты библиотек | Возможны | Невозможны |
| Разные версии torch | Нет | Да |
| Микс local + remote | — | Да |

---

## 10. Docker-сборка

Сборка оптимизирована для быстрой итерации:

- **Базовый образ** `pytorch/pytorch` — torch + CUDA уже внутри
- **uv** вместо pip — установка в 10–100x быстрее
- **Слоистое кэширование** — зависимости в отдельных Docker-слоях
- **BuildKit cache mounts** — пакеты переиспользуются между сборками
- **Non-root user** — контейнер работает от `infergate`

### Скорость пересборки

| Сценарий | Время |
|----------|-------|
| Первая сборка | ~15–20 мин |
| Изменение кода (app/) | ~10 сек |
| Новая зависимость в base.txt | ~1–2 мин |
| Новая ML-зависимость | ~5–10 мин |

---

## 11. Тестирование

```bash
pip install -e ".[dev]"
pytest

# С покрытием
pip install pytest-cov
pytest --cov=app --cov-branch
```

**91 тест**, **82% покрытия** (ветвевое). Покрытие исключает GPU-провайдеры, которые требуют физическое GPU для интеграционного тестирования.

Тесты покрывают: роутеры, middleware (auth, rate-limit, access-log), cache manager, GPU scheduler, provider manager, конфигурацию, валидацию, worker, конкурентность, edge cases.

---

## 12. Системные требования

| Конфигурация | GPU | RAM | Диск |
|-------------|-----|-----|------|
| Минимальная | RTX 3060 12 GB | 16 GB | 50 GB |
| Рекомендуемая | RTX 3090 24 GB | 32 GB | 100 GB |
| Оптимальная | RTX 4090 24 GB | 64 GB | 200 GB |

---

## 13. TODO

- [ ] **Web UI** — панель администрирования
- [ ] **Prometheus метрики** — экспорт метрик + Grafana дашборд
- [ ] **Voice cloning** — клонирование голоса через XTTS-v2 / OpenAudio S1
- [ ] **LoRA hot-swap** — указание LoRA-адаптеров в YAML-конфиге модели
- [ ] **Speech-to-Text** — эндпоинт `/v1/audio/transcriptions`
- [ ] **Multi-GPU** — распределение моделей по нескольким GPU (CUDA device_ids)
- [ ] **Hot-reload конфигов** — добавление моделей без перезапуска сервера
- [ ] **Kubernetes Helm chart** — для multi-node distributed-режима

---

## 14. Лицензия

MIT
