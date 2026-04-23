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
| `stt` | whisper-base (enabled) |
| `upscale` | realesrgan-x4 (enabled) |
| `voice-clone` | xtts-v2 (disabled — CPML license opt-in) |
| `qwen3.5-4b`, `sd35-medium`, `kokoro-82m`, `whisper-base`, `realesrgan-x4`, `xtts-v2` | Индивидуальные |
| `qwen3.5-9b`, `qwen3-8b`, `llama3.1-8b`, `flux1-dev`, `flux1-schnell`, `flux2-klein-4b`, `openaudio-s1-mini`, `xtts-v2` | Disabled по умолчанию, запуск через индивидуальный profile |

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

# Распознавание речи (multipart — как OpenAI /v1/audio/transcriptions)
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@speech.mp3" -F "model=whisper-base" -F "language=ru"

# STT subtitle форматы — srt и vtt
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@speech.mp3" -F "model=whisper-base" -F "response_format=srt" \
  -o subtitles.srt

# Voice cloning (XTTS-v2) — 6-секундный reference + текст → склонированный голос
curl http://localhost:8000/v1/audio/speech/voice-clone \
  -F "reference_audio=@my_voice.wav" -F "input=Hello in my voice" \
  -F "model=xtts-v2" -F "language=en" -o cloned.wav

# Upscale изображения 4×
curl http://localhost:8000/v1/images/upscale \
  -F "file=@small.png" -F "model=realesrgan-x4" -F "response_format=png" \
  -o upscaled.png

# img2img / inpaint через multipart (OpenAI-style /v1/images/edits)
curl http://localhost:8000/v1/images/edits \
  -F "image=@base.png" -F "mask=@mask.png" \
  -F "prompt=replace with a red balloon" -F "model=sdxl-base" \
  -F "denoising_strength=0.8"
```

### Продвинутые примеры генерации изображений

**1. Свой семплер, negative prompt, свои шаги/CFG:**
```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl-base",
    "prompt": "portrait of an astronaut on Mars",
    "negative_prompt": "blurry, low quality, cropped",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "scheduler": "dpm++_2m_karras",
    "seed": 42
  }'
```

**2. Token weighting (A1111-синтаксис, через compel):**
```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl-base",
    "prompt": "(cyberpunk:1.5) cat, (neon lights:0.8) background, (detailed:1.2) fur"
  }'
```
Plain промпт без `(word:weight)` идёт по обычному пути tokenizer'а — никакого regress по качеству.

**3. LoRA hot-load:**
```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl-base",
    "prompt": "a portrait",
    "loras": [
      {"id": "ostris/crayon_style_lora_sdxl", "weight": 0.6},
      {"id": "nerijs/pixel-art-xl", "weight": 0.4}
    ]
  }'
```
До 5 LoRA за запрос. LRU-кэш адаптеров (дефолт 8) — повторные запросы с теми же LoRA не перекачивают веса.

**4. Textual Inversion:**
```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl-base",
    "prompt": "<s0><s1> a portrait photo",
    "textual_inversions": [
      {"id": "linoyts/web_y2k",
       "token": ["<s0>", "<s1>"],
       "weight_file": "web_y2k_emb.safetensors"}
    ]
  }'
```
Для multi-token pivotal-embeddings (SDXL) провайдер автоматически определяет dual-tensor формат и регистрирует в clip_l и clip_g отдельно.

**5. HighresFix (двухпроходная генерация):**
```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl-base",
    "prompt": "detailed fantasy landscape",
    "size": "768x768",
    "highres_fix": {
      "scale": 1.5,
      "denoising_strength": 0.5,
      "upscaler": "lanczos"
    }
  }'
```
Выход: 1152×1152 (768 × 1.5). Первый проход генерит 768 → PIL-upscale → img2img refine. Меньше duplicate-артефактов чем single-pass на той же конечной резолюции.

**6. Комбо — всё сразу:**
```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl-base",
    "prompt": "(cyberpunk:1.3) cat",
    "negative_prompt": "low quality",
    "scheduler": "dpm++_2m_karras",
    "num_inference_steps": 30,
    "seed": 42,
    "loras": [{"id": "ostris/crayon_style_lora_sdxl", "weight": 0.7}],
    "highres_fix": {"scale": 1.5, "denoising_strength": 0.4}
  }'
```
Порядок применения внутри провайдера: scheduler swap → LoRA load/activate → TI register → compel encoding → pipeline call (с HighresFix двумя проходами если указан).

---

## 3. Поддерживаемые модели

### Генерация изображений

| Модель | Архитектура | Провайдер | VRAM | Разрешение | Лицензия |
|--------|-------------|-----------|------|-----------|----------|
| **Stable Diffusion 3.5 Medium** | MMDiT | diffusers | 5 GB (drop_t5) | 1024 | Community |
| **Stable Diffusion XL Base 1.0** | UNet | diffusers | 7 GB | 1024 | OpenRAIL++ |
| **FLUX.1 Schnell** | MMDiT (flow) | diffusers | 9 GB (nf4) | 1024 | Apache 2.0 |
| **FLUX.1 Dev** | MMDiT (flow) | diffusers | 12 GB | 1024 | Non-Commercial |
| **FLUX.2 Klein 4B** | DiT | diffusers | ~14 GB | 1024 | Apache 2.0 |
| **Qwen-Image** | DiT | diffusers | 11 GB (nf4+offload) | 1024 | Apache 2.0 |
| **Hunyuan-DiT v1.2** | DiT | diffusers | 7 GB | 1024 | Tencent Community |
| **Z-Image Turbo** | DiT (distilled) | diffusers | 11 GB | 1024 | Apache 2.0 |
| **Janus-Pro 1B** | **Autoregressive** | custom | 3 GB | 384 | MIT |
| **Janus-Pro 7B** | **Autoregressive** (nf4) | custom | 7 GB | 384 | MIT |
| **Meissonic** | **Masked non-AR** | custom (vendored) | 10 GB | 1024 | Apache 2.0 |

### Текст, озвучка, распознавание, super-resolution

| Модель | Категория | Провайдер | VRAM | Лицензия |
|--------|-----------|-----------|------|----------|
| **Qwen 3.5 4B (AWQ)** | Текст | vLLM | 4 GB | Apache 2.0 |
| **Qwen 3.5 9B** | Текст | vLLM | 8 GB | Apache 2.0 |
| **Qwen 3 8B** | Текст | vLLM | 7 GB | Apache 2.0 |
| **Llama 3.1 8B** | Текст | vLLM | 8 GB | Llama Community |
| **Kokoro 82M** | TTS | kokoro | CPU | MIT |
| **OpenAudio S1 Mini** | TTS | fish-speech | 4 GB | Apache 2.0 |
| **XTTS v2** | Voice-cloning TTS | coqui-tts | 2 GB (fp16) | CPML (non-commercial) |
| **Whisper Base** | STT | faster-whisper (CT2) | CPU (int8, ~90 MB) | MIT |
| **Real-ESRGAN 4×** | Upscale | spandrel | 1.5 GB (fp16) | BSD-3-Clause |

**Три архитектурных семейства изображений** в одном стеке: диффузия (SD/FLUX/Qwen/Hunyuan/Z-Image), autoregressive (Janus-Pro) и masked non-AR (Meissonic) — все под единым `ImageProvider` интерфейсом с OpenAI-совместимым API. STT/Upscale/TTS живут под собственными ABC-категориями (`SttProvider`, `ImageUpscaleProvider`, `TtsProvider`).

Любая diffusers-совместимая модель добавляется одним YAML-файлом без написания кода. Для нестандартных архитектур (AR / Masked / другое) — добавляется новый класс-провайдер, автоматически регистрируемый через `@register_provider`.

---

## 4. API

Все эндпоинты совместимы с форматом OpenAI API.

### Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/v1/chat/completions` | Генерация текста (+ streaming) |
| `POST` | `/v1/images/generations` | Генерация изображений (+ img2img/inpaint через base64) |
| `POST` | `/v1/images/edits` | img2img / inpaint (multipart, OpenAI-style) |
| `POST` | `/v1/audio/speech` | Синтез речи (JSON) |
| `POST` | `/v1/audio/speech/voice-clone` | Voice-cloning (multipart, reference audio) |
| `POST` | `/v1/audio/transcriptions` | Распознавание речи (multipart; `json`/`text`/`verbose_json`/`srt`/`vtt`) |
| `POST` | `/v1/images/upscale` | Super-resolution изображений (multipart) |
| `GET` | `/v1/models` | Список всех моделей |
| `POST` | `/v1/models/{id}/load` | Загрузить модель в GPU |
| `POST` | `/v1/models/{id}/unload` | Выгрузить модель |
| `GET` | `/cache/stats` | Статистика кэша |
| `GET` | `/cache/stats/{model_id}` | Статистика кэша одной модели |
| `DELETE` | `/cache` | Очистить весь кэш |
| `DELETE` | `/cache/{model_id}` | Очистить кэш модели |
| `DELETE` | `/cache/entry/{key}` | Удалить запись кэша |
| `GET` | `/health` | Проверка состояния |
| `GET` | `/metrics` | Метрики системы (JSON snapshot) |
| `GET` | `/metrics/prometheus` | Prometheus scrape endpoint (для Grafana) |

### Поля запроса для генерации изображений

Все поля опциональны кроме `prompt`. Неуказанные значения берутся из YAML-defaults модели.

| Поле | Тип | Диапазон | Описание |
|------|-----|----------|----------|
| `model` | str | — | ID модели (если не указан, берётся default категории) |
| `prompt` | str | 1-10000 | Основной промпт. Поддерживает `(word:1.5)`-синтаксис через compel |
| `negative_prompt` | str | 0-10000 | Отрицательный промпт |
| `size` | str | — | `WxH`, например `1024x1024` |
| `seed` | int | — | Детерминированная генерация |
| `n` | int | 1-10 | Количество картинок |
| `num_inference_steps` | int | 1-150 | Количество шагов denoising |
| `guidance_scale` | float | 0.0-30.0 | Сила следования промпту (CFG) |
| `scheduler` | str | см. ниже | Семплер: `euler`, `euler_a`, `dpm++_2m`, `dpm++_2m_karras`, `dpm++_sde`, `ddim`, `ddpm`, `lms`, `heun`, `pndm`, `unipc` |
| `loras` | list | ≤5 | `[{"id": "user/repo", "weight": 0.8, "weight_file"?: "...", "adapter_name"?: "..."}]` |
| `textual_inversions` | list | ≤10 | `[{"id": "user/repo", "token"?: "str\|list", "weight_file"?: "..."}]` |
| `highres_fix` | object | — | `{"scale": 1.5, "denoising_strength": 0.5, "steps"?: int, "upscaler": "lanczos"}` |

### Поля для текста (chat completions)

| Параметр | Диапазон |
|----------|----------|
| `temperature` | 0.0 – 2.0 |
| `top_p` | 0.0 – 1.0 |
| `max_tokens` | 1 – 131072 |
| `stream` | bool (SSE) |

### Поля для озвучки

| Параметр | Диапазон |
|----------|----------|
| `speed` | 0.25 – 4.0 |
| `voice` | str (зависит от модели) |

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

### Memory safety — три рубежа защиты

Runaway-модель не должна взорвать хост. Защита многослойная, каждый слой параметризуется:

**Слой 1: docker container limits** (`deploy/.env`)
Linux OOMKiller убивает контейнер прежде, чем хост уйдёт в swap-spiral.
```
GPU_WORKER_MEM_LIMIT=16g    # RAM-cap на GPU-воркер
CPU_WORKER_MEM_LIMIT=6g     # RAM-cap на CPU-воркер (kokoro, whisper)
GATEWAY_MEM_LIMIT=2g        # RAM-cap на gateway
GPU_WORKER_SHM_SIZE=2g      # /dev/shm для torch multiprocessing
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
```

**Слой 2: byte-budget LRU** (`config/server.yaml`, env-override)
`ProviderManager` вытесняет LRU когда сумма объявленных `vram_mb` загруженных моделей + новая модель > бюджет. Если вытеснять нечего (всё pinned), возвращается **HTTP 503 `insufficient_resources`** вместо CUDA OOM.
```
GPU_MAX_VRAM_BUDGET_MB=10000   # 12GB → 10000, 24GB → 22000, 48GB → 44000
GPU_VRAM_HEADROOM_MB=2000      # Резерв под activation spikes
```

**Слой 3: MemoryWatchdog** (фоновая задача)
Опрашивает `GET /stats` на каждом загруженном воркере (`torch.cuda.mem_get_info` + `psutil.virtual_memory`). При превышении порога — аварийно вытесняет LRU. Ловит леаки и подзанижения `vram_mb` в YAML'ах, которые статический бюджет не видит.
```
GPU_WATCHDOG_INTERVAL_SECONDS=15   # 0 = off (default)
GPU_WATCHDOG_VRAM_THRESHOLD=0.92   # evict LRU когда live VRAM ≥ 92%
GPU_WATCHDOG_RAM_THRESHOLD=0.90    # warn (без eviction) для host RAM
```

Метрики живого VRAM/RAM доступны на `GET /v1/models` (per-model) и через воркерский `GET /stats` (live). Prometheus дашборд рендерит их в панели «GPU VRAM used».

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
  max_loaded_models: 3             # Backstop count-based LRU
  pinned_models: []                # Модели, которые не выгружаются
  # Primary: byte-budget LRU. When sum(declared vram_mb of loaded models)
  # would exceed this, evict LRU until it fits. 0 = disabled.
  # 12GB → 10000, 24GB → 22000, 48GB → 44000.
  max_vram_budget_mb: 0
  vram_headroom_mb: 0              # Safety headroom for activation spikes
  # MemoryWatchdog — live VRAM/RAM monitor; emergency-evict LRU at threshold.
  # 0 = disabled; 10-30s recommended in production.
  watchdog_interval_seconds: 0
  watchdog_vram_threshold: 0.92
  watchdog_ram_threshold: 0.90

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
│   │   ├── images.py               # /v1/images/{generations,edits,upscale}
│   │   ├── audio.py                # /v1/audio/{speech,speech/voice-clone,transcriptions}
│   │   ├── models.py               # /v1/models
│   │   ├── cache.py                # /cache/*
│   │   └── health.py               # /health, /metrics, /metrics/prometheus
│   ├── schemas/                    # Pydantic-модели с Field-валидацией
│   ├── providers/
│   │   ├── base.py                 # ABC: Image/Text/Tts/Stt/ImageUpscale Providers
│   │   ├── registry.py             # @register_provider + авто-обнаружение
│   │   ├── remote.py               # RemoteProvider для distributed-режима (5 категорий)
│   │   ├── image/
│   │   │   ├── diffusers_provider.py  # Diffusion (SD, SDXL, FLUX, Hunyuan, Z-Image, Qwen)
│   │   │   │                         # + LoRA / TI / compel / scheduler swap
│   │   │   │                         # + HighresFix / img2img / inpaint
│   │   │   ├── janus_provider.py     # Autoregressive (DeepSeek Janus-Pro 1B / 7B)
│   │   │   └── meissonic_provider.py # Masked non-AR (vendored Meissonic pipeline)
│   │   ├── text/
│   │   │   └── vllm_provider.py    # + streaming + tokenizer chat templates
│   │   ├── tts/
│   │   │   ├── kokoro.py
│   │   │   └── fish_speech.py      # OpenAudio S1 / Fish Speech + voice-clone stub
│   │   ├── stt/
│   │   │   └── whisper_provider.py # faster-whisper (CTranslate2)
│   │   └── upscale/
│   │       └── spandrel_provider.py # spandrel (ESRGAN / Real-ESRGAN / SwinIR / …)
│   ├── services/
│   │   ├── provider_manager.py     # LRU, per-model locks, reload_model, worker-monitor
│   │   ├── gpu_scheduler.py        # per-model concurrency + priority queue
│   │   ├── cache_manager.py        # SQLite WAL, atomic writes, miss tracking
│   │   └── config_watcher.py       # polls config/models/*.yaml → hot-reload
│   └── worker.py                   # Standalone FastAPI worker (один процесс = одна модель)
│                                   # /generate, /synthesize, /transcribe, /upscale,
│                                   # /voice-clone, /reload, /health, /load, /unload
├── config/
│   ├── server.yaml
│   └── models/                     # 1 YAML = 1 модель (17 файлов)
├── deploy/
│   ├── Dockerfile.gateway          #   Лёгкий gateway (~500MB)
│   ├── Dockerfile.worker           #   Единый параметризованный Dockerfile для всех воркеров
│   ├── docker-compose.yml          #   Compose с profiles (gateway + 21 workers, YAML-якоря)
│   ├── docker-bake.hcl             #   Матрица сборки (1 строка = 1 модель)
│   ├── .env.example
│   ├── docker-compose.server.example.yml
│   ├── workers/                    #   Только requirements.txt per-model
│   │   ├── qwen3.5-4b/requirements.txt
│   │   ├── sd35-medium/requirements.txt
│   │   ├── kokoro-82m/requirements.txt
│   │   ├── whisper-base/requirements.txt
│   │   ├── realesrgan-x4/requirements.txt
│   │   ├── xtts-v2/requirements.txt
│   │   └── …                       #   21 модель
│   └── monitoring/                 #   Prometheus + Grafana stack (auto-provisioning)
│       ├── prometheus.yml
│       ├── docker-compose.monitoring.yml
│       └── grafana/                #   datasource + dashboards provisioning
├── scripts/
│   ├── diagnose/                   #   Per-model smoke-scripts
│   └── feature/                    #   End-to-end feature tests (на живых воркерах)
├── tests/                          #   194 pytest-ов
└── pyproject.toml
```

---

## 9. Per-model архитектура

Каждая модель работает в изолированном контейнере с собственным Dockerfile и зависимостями. Gateway — лёгкий образ (~500MB) без ML-библиотек.

```
Client → Gateway (500MB, без GPU)
           ├→ worker-qwen3-5-4b       (vLLM, GPU)           — text
           ├→ worker-sd35-medium      (diffusers, GPU)      — image
           ├→ worker-kokoro-82m       (kokoro, CPU)         — TTS
           ├→ worker-xtts-v2          (coqui-tts, GPU)      — voice cloning
           ├→ worker-whisper-base     (faster-whisper, CPU) — STT
           ├→ worker-realesrgan-x4    (spandrel, GPU)       — upscale
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

Grafana поднимается с auto-provisioning: Prometheus datasource (`infergate-prometheus`) и готовый дашборд **InferGate — Gateway & Inference** (папка «InferGate») подключаются автоматически — клик-через через UI не нужен. Сам дашборд лежит в `deploy/monitoring/grafana/dashboards/infergate.json` и содержит панели для request rate/latency/error-rate, inference p95 per-model, cache hit-ratio per-model и стат-плашки для VRAM / queue / loaded-models.

`prometheus-client` входит в `requirements/base.txt` (gateway), поэтому `/metrics/prometheus` работает out-of-box. Модуль `app/monitoring/metrics.py` сохраняет graceful-degradation (501 Not Implemented), если кто-то соберёт минимальный образ без него. В `pyproject.toml` extra `monitoring` также остаётся — для pip-установок без Docker: `pip install infergate[monitoring]`.

---

## 11. Docker-сборка

Сборка построена на принципе **одно описание — много вариантов**:

- **Единый `deploy/Dockerfile.worker`** — параметризован build-args (`BASE_IMAGE`, `APT_PACKAGES`, `WORKER_REQUIREMENTS`, `POST_INSTALL`). Все 21 воркеров (image / text / tts / stt / upscale / voice-clone) собираются из него — нет копипаста.
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

**194 pytest-а** (исключая GPU-провайдеры — они проверяются e2e-скриптами против реальных воркеров, см. `scripts/feature/`).

Pytest покрывает: роутеры (`chat`, `images`, `audio`, `models`, `cache`, `health`), middleware (auth, rate-limit, access-log), мониторинг (Prometheus-лейблы, request ID), `CacheManager`, `GpuScheduler`, `ProviderManager` (включая hot-reload `reload_model`), `ConfigWatcher`, воркер-эндпоинты (`/generate`, `/synthesize`, `/transcribe`, `/upscale`, `/reload`, `/voice-clone`), конфигурацию, валидацию схем, конкурентность, edge cases.

**E2e feature-скрипты** (`scripts/feature/*.sh`) — прогоняются на живых Docker-контейнерах и закрывают то, что нельзя через моки: `sd35-lora.sh`, `img2img.sh`, `hot-reload-config.sh`, `worker-reload.sh`, `stt-whisper.sh`, `upscale.sh`, `voice-clone-xtts.sh`, `grafana-provisioning.sh`, `lora-hot-load.sh`, `textual-inversion.sh`, `highres-fix.sh`, `scheduler-swap.sh`, `token-weighting.sh`, `per-request-tunables.sh`, `error-forwarding.sh`.

---

## 13. Системные требования

| Конфигурация | GPU | RAM | Диск |
|-------------|-----|-----|------|
| Минимальная | RTX 3060 12 GB | 16 GB | 50 GB |
| Рекомендуемая | RTX 3090 24 GB | 32 GB | 100 GB |
| Оптимальная | RTX 4090 24 GB | 64 GB | 200 GB |

---

## 14. TODO

Список того, что ещё предстоит сделать — в порядке приоритета.

### Фичи

1. [ ] **SDXL Refiner** — ensemble `base → refiner` с передачей latents. `SDXLRefinerImageProvider` должен держать две модели одновременно в VRAM (~14 GB суммарно); на 12 GB картах — sequential unload/load base→refiner или CPU offload обоих. Требует нового провайдера или custom-логики в `DiffusersImageProvider`.
2. [ ] **Multi-GPU** — per-worker `CUDA_VISIBLE_DEVICES`-routing и distributed-LRU в `ProviderManager`. Нужно для multi-GPU машин, где сейчас все воркеры по умолчанию борются за первый GPU.
3. [ ] **Upscaler tiling** — follow-up к `/v1/images/upscale`. Сейчас вход ограничен `max_input_side=2048` (guard против OOM). Для upscale изображений большего размера нужно tile-разбиение с перекрытием + gradient blending в `SpandrelUpscaleProvider`.
4. [ ] **Kubernetes Helm chart** — `deploy/helm/` с шаблонами Deployment (gateway + per-worker), ConfigMap для YAML, PVC для `models/` (веса), HPA. Для multi-node development/production.
5. [ ] **Web UI** — админ-панель: gallery сгенерированного, история prompt-ов, live-метрики (уже есть JSON `/metrics` и Prometheus — остаётся frontend).

### Рефакторинг и code-hygiene

6. [x] **Разбить `diffusers_provider.py`** на подмодули: `_compel.py`, `_lora.py`, `_textual_inversion.py`, `_highres_fix.py`, `_schedulers.py`. Провайдер остался фасадом (load/unload/generate), каждая вертикаль изолирована в своём модуле.
7. [x] **Общий `BaseRemoteMixin` для `RemoteProvider`**. Транспорт (`load/unload/check_health/reload/get_stats`) вынесен в миксин; подклассы (`RemoteText/Image/Tts/Stt/Upscale`) оставили только per-категорию сериализацию.
8. [x] **Prometheus-гейджи** — инициализация и публикация через `update_runtime_gauges()` в `app/monitoring/metrics.py`. Роутер `/metrics` только вычисляет значения и делегирует.
9. [x] **Логгеры централизованы на `logging.getLogger(__name__)`** во всех модулях `app/` (убраны именованные `"infergate"`, `"infergate.worker"`, `"infergate.access"`).
10. [ ] **Включить ruff в CI как блокирующую проверку**. `ruff check tests/` уже чистый (было 26 ошибок импорт-ордера / unused — все исправлены), осталось завести CI-конфиг (`.github/workflows/ci.yml` или аналог).

---

## 15. Лицензия

MIT
