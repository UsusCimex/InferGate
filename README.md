# InferGate

Self-hosted OpenAI-совместимый шлюз к локальным моделям: генерация и правка картинок, текст (в том числе с картинками во входе), озвучка и клонирование голоса, распознавание речи, апскейл, эмбеддинги. Клиент OpenAI API переключается на InferGate сменой `base_url`.

Каждая модель работает в своём Docker-контейнере со своими зависимостями. Шлюз на FastAPI загружает модели по требованию, вытесняет давно не использованные по бюджету видеопамяти, ставит запросы в очередь и кэширует ответы. Основной клиент: Android-приложение PictoLex.

## Модели

| Категория | Модели |
|---|---|
| Картинки | SDXL, SD 3.5 Medium, FLUX.1 Schnell и Dev, FLUX.2 Klein 4B, Qwen-Image, Hunyuan-DiT, Z-Image Turbo (диффузия и flow matching), Janus-Pro 1B и 7B (авторегрессия), Meissonic (маскированная генерация) |
| Текст (vLLM) | Qwen 3.5 4B (с картинками во входе), Qwen 3.5 9B, Qwen 3 8B, Llama 3.1 8B |
| Озвучка | Kokoro 82M, VoxCPM2 (четыре рассказчика и клонирование голоса), Qwen3-TTS 0.6B и XTTS v2 (клонирование), OpenAudio S1 Mini |
| Распознавание | Whisper Base (faster-whisper) |
| Апскейл | Real-ESRGAN x4 |
| Эмбеддинги | Multilingual E5, CLIP, SigLIP, CLAP, CLIP4Clip |

Модель с существующим провайдером добавляется YAML-файлом, строкой в матрице сборки и сервисом Compose, без кода Python. Каталог, профили и VRAM: [docs/models.md](docs/models.md).

## Быстрый старт

```bash
git clone https://github.com/UsusCimex/InferGate.git
cd InferGate
cp deploy/.env.example deploy/.env       # HF_TOKEN, COMPOSE_PROFILES
docker compose -f deploy/docker-compose.yml --profile text --profile tts up -d
curl http://localhost:8000/health
```

Профиль: категория (`text`, `image`, `tts`, `stt`, `upscale`, `voice-clone`, `embedding`) или id модели (`flux2-klein-4b`). Умолчания в `config/models/*.yaml` рассчитаны на видеокарту 12 ГБ, под другое железо меняются переменные в `deploy/.env`. Swagger: `http://localhost:8000/docs`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")
client.chat.completions.create(model="qwen3.5-4b", messages=[{"role": "user", "content": "Привет!"}])
client.images.generate(model="flux2-klein-4b", prompt="A red kite in the sky")
client.audio.speech.create(model="kokoro-82m", input="Hello, world!", voice="af_heart")
```

## Документация

| Документ | О чём |
|---|---|
| [docs/api.md](docs/api.md) | эндпоинты, поля запросов, заголовки, ошибки, примеры |
| [docs/models.md](docs/models.md) | каталог моделей, профили Compose, провайдеры, добавление модели |
| [docs/configuration.md](docs/configuration.md) | `server.yaml`, YAML модели, переменные `deploy/.env`, защита памяти |
| [docs/architecture.md](docs/architecture.md) | шлюз и воркеры, путь запроса, менеджер моделей, очередь, кэш |
| [docs/deployment.md](docs/deployment.md) | Docker Compose, сборка образов, HTTPS, мониторинг, веб-страница, Kubernetes, требования к железу |
| [docs/testing.md](docs/testing.md) | pytest, CI, проверки на живых контейнерах |
| [CONTRIBUTING.md](CONTRIBUTING.md) | правила задач, коммитов, веток и текстов |
| [CLAUDE.md](CLAUDE.md) | правила для Claude Code |
| [Issues](https://github.com/UsusCimex/InferGate/issues) | задачи и известные проблемы |

## Разработка

```bash
pip install -e ".[dev]"
pytest
ruff check app/ tests/
uvicorn app.main:app --reload
```

Python 3.11+. Правила работы: [CONTRIBUTING.md](CONTRIBUTING.md), стиль кода: [CLAUDE.md](CLAUDE.md).

## Лицензия

MIT. Лицензии моделей: [docs/models.md](docs/models.md).
