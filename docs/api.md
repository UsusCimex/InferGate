# API

Шлюз слушает порт 8000. Эндпоинты `/v1/chat/completions`, `/v1/images/generations`, `/v1/images/edits`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/embeddings` и `/v1/models` повторяют формат OpenAI, поэтому OpenAI SDK работает со шлюзом после смены `base_url`. Остальные — расширения InferGate.

Если `model` не указан, берётся модель по умолчанию из `defaults` в `config/server.yaml` (для распознавания и апскейла умолчаний нет — `model` обязателен). Все JSON-схемы запрещают лишние поля: неизвестное поле — ответ 422 с его именем.

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/v1/chat/completions` | текст, потоковый вывод (SSE), картинки во входе для моделей с `capabilities.vision` |
| `POST` | `/v1/images/generations` | генерация картинок; img2img и inpaint — полями `image`/`mask` (base64) |
| `POST` | `/v1/images/edits` | img2img и inpaint, multipart |
| `POST` | `/v1/images/upscale` | увеличение разрешения, multipart |
| `POST` | `/v1/audio/speech` | синтез речи |
| `POST` | `/v1/audio/speech/voice-clone` | синтез клонированным голосом, multipart |
| `POST` | `/v1/audio/transcriptions` | распознавание речи, multipart |
| `POST` | `/v1/embeddings` | эмбеддинги текста |
| `POST` | `/v1/embeddings/audio`, `/v1/embeddings/image`, `/v1/embeddings/video` | эмбеддинг одного файла, multipart |
| `GET` | `/v1/models` | все модели: категория, включена ли, загружена ли, метаданные и теги |
| `POST` | `/v1/models/{id}/load`, `/v1/models/{id}/unload` | загрузить или выгрузить модель |
| `GET` | `/v1/admin/memory/status` | загруженные модели, объявленная VRAM, бюджет, закреплённые модели |
| `GET` | `/v1/admin/memory/preview-load/{id}` | план вытеснения, если загрузить модель |
| `GET` | `/cache/stats`, `/cache/stats/{model_id}` | статистика кэша |
| `DELETE` | `/cache`, `/cache/{model_id}`, `/cache/entry/{key}` | очистка кэша |
| `GET` | `/health`, `/v1/health` | `{"status": "ok"}`; состояние воркеров не проверяется |
| `GET` | `/metrics` | JSON-снимок: очередь, загруженные модели, доля попаданий в кэш, uptime |
| `GET` | `/metrics/prometheus` | метрики для Prometheus |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | документация FastAPI |

## Текст — `/v1/chat/completions`

| Поле | Ограничения |
|---|---|
| `messages` | 1–500 сообщений `{role, content}`; `content` — строка или список частей `{"type": "text", "text"}` и `{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}` |
| `temperature` | 0–2 |
| `top_p` | 0–1 |
| `max_tokens` | 1–131072 |
| `stream` | SSE-поток, заканчивается `data: [DONE]` |
| `response_format` | `{"type": "json_object"}` добавляет системную инструкцию «отвечай только JSON»; управляемой генерации нет |
| `thinking` | расширение: включает рассуждения модели (по умолчанию — из YAML модели); при включённом `max_tokens` поднимается минимум до 4096 |

Картинки принимаются только как `data:`-URL и только моделями с `capabilities.vision: true` (иначе 400 `vision_not_supported`); число картинок на запрос ограничено `limit_mm_per_prompt` модели. Ответ возвращается как есть: рассуждения не отделяются, `finish_reason` всегда `stop`. Поля OpenAI `stop`, `n`, `seed`, `tools`, `presence_penalty` не поддерживаются (422).

## Картинки

**`/v1/images/generations`** — обязательно только `prompt`; остальное берётся из `default_params` модели.

| Поле | Ограничения | Назначение |
|---|---|---|
| `prompt` | 1–10000 | промт; синтаксис весов `(word:1.5)` работает на sdxl-base (compel), у остальных моделей веса снимаются |
| `negative_prompt` | ≤10000 | пустой считается отсутствующим |
| `size` | `WxH` | приоритетнее ширины и высоты из YAML |
| `n` | 1–10 | |
| `seed` | целое | включает кэш для моделей со стратегией `seed_only` |
| `num_inference_steps` | 1–150 | |
| `guidance_scale` | 0–30 | |
| `scheduler` | `euler`, `euler_a`, `dpm++_2m`, `dpm++_2m_karras`, `dpm++_sde`, `ddim`, `ddpm`, `lms`, `heun`, `pndm`, `unipc` | модели на flow matching отвечают 400 |
| `loras` | ≤5 `{id, weight, weight_file?, adapter_name?}` | горячая загрузка LoRA с Hugging Face (нужен `peft` — есть в воркерах sdxl-base и sd35-medium) |
| `textual_inversions` | ≤10 `{id, token?, weight_file?}` | |
| `highres_fix` | `{scale (1, 4], denoising_strength, steps?, upscaler}` | двухпроходная генерация |
| `image`, `mask` | base64 | img2img и inpaint |
| `denoising_strength` | 0–1 | сила img2img |
| `refiner_switch_at` | 0–1 | доля шагов базы перед SDXL Refiner (нужен `refiner_hub_id` модели) |
| `response_format` | `b64_json` или `url` (data URL) | |

Порядок применения внутри провайдера: смена планировщика → LoRA → Textual Inversion → seed → веса compel → генерация (HighresFix, img2img/inpaint, Refiner или обычный вызов).

**`/v1/images/edits`** (multipart): `image` (обязательно), `prompt` (обязательно), `mask?`, `model?`, `n`, `size` (по умолчанию `1024x1024`), `response_format`, `seed`, `negative_prompt`, `num_inference_steps`, `guidance_scale`, `scheduler`, `denoising_strength`.

**`/v1/images/upscale`** (multipart): `file`, `model` (например `realesrgan-x4`), `response_format` — `b64_json` или `png` (тогда ответ — сам PNG).

## Аудио

**`/v1/audio/speech`** (JSON):

| Поле | Ограничения |
|---|---|
| `input` | 1–100000 символов |
| `voice` | по умолчанию `default` — голос модели; модели со списком `capabilities.voices` (voxcpm2) отвечают 400 на другой голос и возвращают список в `error.voices` |
| `response_format` | `mp3` (по умолчанию), `wav`, `flac`, `opus` |
| `speed` | 0.25–4 |
| `language` | ≤32; имя языка для qwen3-tts (`English`, `Russian`, …), код для xtts; одноязычные модели его игнорируют |
| `seed` | 0–2³²−1, входит в ключ кэша; voxcpm2 без него берёт 42, поэтому повтор совпадает байт в байт |

Модели с `capabilities.voice_clone_only` (qwen3-tts-06b, xtts-v2) на этом эндпоинте отвечают 400 `voice_clone_required`.

**`/v1/audio/speech/voice-clone`** (multipart): `reference_audio` (файл), `input`, `model?`, `reference_text?`, `response_format`, `speed`, `language`, `seed`. С `reference_text` voxcpm2 продолжает референс, без него берёт только тембр; qwen3-tts требует `reference_text`; xtts его игнорирует.

**`/v1/audio/transcriptions`** (multipart): `file`, `model` (например `whisper-base`), `language?`, `prompt?`, `temperature` (0–1), `vad_filter?`, `response_format` — `json` (`{text}`), `text`, `verbose_json` (`{text, language, duration, segments}`), `srt`, `vtt` (последние два собираются шлюзом из `verbose_json`).

## Эмбеддинги

- `/v1/embeddings` (JSON): `input` — строка или список строк, `encoding_format` — только `float`. Ответ в формате OpenAI; `usage` всегда нулевой.
- `/v1/embeddings/audio`, `/image`, `/video` (multipart): один `file` и `model?`; ответ `{model, embedding}`.

Модели эмбеддингов (E5, CLAP, CLIP, SigLIP, CLIP4Clip) по умолчанию выключены: нужно `<ID>_ENABLED=true` в `deploy/.env` и профиль `embedding` (SigLIP — только своим профилем). Векторы L2-нормированы, кэш не используется.

## Заголовки

Запрос:

| Заголовок | Назначение |
|---|---|
| `Authorization: Bearer <key>` | при включённой авторизации (`auth.enabled` и непустой `api_keys`) |
| `X-InferGate-No-Cache: true` | не брать ответ из кэша |
| `X-Request-ID` | id запроса для журналов; без него шлюз создаёт свой и передаёт воркеру |

Ответ:

| Заголовок | Значение |
|---|---|
| `X-InferGate-Cache` | `HIT`, `MISS`, `DISABLED`, `SKIP` |
| `X-InferGate-Model` | модель, которая ответила |
| `X-InferGate-Generation-Ms` | время запроса в миллисекундах |
| `X-InferGate-Load-Ms`, `X-InferGate-Inference-Ms` | загрузка модели и сам вывод (текст) |
| `X-InferGate-Queue-Position` | число запросов в работе при последней постановке в очередь (это не позиция этого запроса) |
| `X-Request-ID` | id запроса |
| `Retry-After` | при 429 |

## Ошибки

Тело ошибки: `{"error": {"message", "type", ...}}`.

| Статус | `type` |
|---|---|
| 400 | `vision_not_supported`, `invalid_image`, `voice_clone_required`, `invalid_request` |
| 401 | `authentication_error` |
| 404 | `not_found` |
| 413 | `upload_too_large` (лимиты `upload_limits`: картинка 20 МБ, аудио 25 МБ, видео 200 МБ, апскейл 50 МБ) |
| 422 | `invalid_request` — ошибка валидации, `param` называет поле |
| 429 | `rate_limit_exceeded` |
| 503 | `worker_not_ready` (контейнер модели не запущен), `insufficient_resources` (модель не помещается в бюджет VRAM), `queue_full`, `model_not_ready` |
| 504 | `timeout` |

Ошибки воркера, которые шлюз не распознал, передаются как есть (`upstream_error`). Часть ошибок роутеров приходит без `type`.

## Примеры

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Привет!"}],
)
client.images.generate(model="flux2-klein-4b", prompt="A red kite in the sky")
client.audio.speech.create(model="kokoro-82m", input="Hello, world!", voice="af_heart")
```

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-4b", "messages": [{"role": "user", "content": "Привет!"}]}'

curl http://localhost:8000/v1/images/generations -H "Content-Type: application/json" \
  -d '{"model": "sdxl-base", "prompt": "(cyberpunk:1.3) cat", "negative_prompt": "low quality",
       "scheduler": "dpm++_2m_karras", "num_inference_steps": 30, "seed": 42,
       "loras": [{"id": "ostris/crayon_style_lora_sdxl", "weight": 0.7}],
       "highres_fix": {"scale": 1.5, "denoising_strength": 0.4}}'

curl http://localhost:8000/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model": "voxcpm2", "input": "Once upon a time...", "voice": "vox_clara"}' -o tale.mp3

curl http://localhost:8000/v1/audio/speech/voice-clone \
  -F "reference_audio=@speaker.wav" -F "reference_text=Exact words of the clip" \
  -F "input=Hello in my voice" -F "model=voxcpm2" -o cloned.mp3

curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@speech.mp3" -F "model=whisper-base" -F "response_format=srt" -o subtitles.srt

curl http://localhost:8000/v1/images/upscale \
  -F "file=@small.png" -F "model=realesrgan-x4" -F "response_format=png" -o upscaled.png

curl http://localhost:8000/v1/images/edits \
  -F "image=@base.png" -F "mask=@mask.png" -F "prompt=a red balloon" \
  -F "model=sdxl-base" -F "denoising_strength=0.8"
```

Известные расхождения с OpenAI и ограничения перечислены в [roadmap.md](roadmap.md#известные-проблемы).
