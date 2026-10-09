# Тесты и проверки

## pytest

```bash
pip install -e ".[dev]"
pytest                          # без GPU и сети
pytest tests/test_chat.py
ruff check app/ tests/          # как в CI
pytest --cov=app --cov-branch   # покрытие, нужен pytest-cov
```

Тесты асинхронные (`asyncio_mode=auto`). `tests/conftest.py` даёт поддельные провайдеры всех категорий (`FakeImageProvider`, `FakeTextProvider`, `FakeTtsProvider` и др.), фикстуру `services` (менеджер моделей, планировщик и кэш на этих провайдерах) и фикстуру `client` (асинхронный httpx-клиент к приложению со `services` в `app.state`, без middleware).

| Что | Файлы |
|---|---|
| роутеры и схемы | `test_chat*.py`, `test_images.py`, `test_audio.py`, `test_embeddings.py`, `test_cache.py`, `test_health_metrics.py`, `test_schema_forbid.py`, `test_upload_limits.py`, `test_tts_voices.py`, `test_voice_clone_only.py` |
| сервисы | `test_provider_manager.py` (вытеснение, бюджет, монитор воркеров, перезагрузка), `test_concurrency.py`, `test_cache_manager.py`, `test_cache_backends.py` (локальный и Redis через fakeredis), `test_config*.py`, `test_memory_watchdog.py` |
| шлюз и воркер | `test_remote_e2e.py` (`RemoteProvider` против поддельного воркера через ASGITransport), `test_remote_provider.py`, `test_worker.py` (настоящие обработчики `app.worker`) |
| провайдеры без GPU | `test_diffusers_provider.py` (два теста пропускаются без torch), `test_weight_syntax.py`, `test_lora_cache.py`; `test_voxcpm2_provider.py` пропускается без numpy, soundfile и pyloudnorm |
| middleware и метрики | `test_middleware.py`, `test_monitoring.py` |

CI (`.github/workflows/ci.yml`): push в `main` и PR, Python 3.11 и 3.12, `ruff check app/ tests/` и `pytest -q`.

## Проверки на живых контейнерах

Bash-скрипты против запущенных воркеров, нужны Docker и `deploy/.env`.

- **`scripts/feature/*.sh`**: LoRA и горячая загрузка, Textual Inversion, веса compel, смена планировщика, HighresFix, SDXL Refiner, img2img, параметры на запрос, апскейл, распознавание речи, озвучка VoxCPM2, клонирование голоса (XTTS, Qwen3-TTS), перезагрузка конфига и воркера, передача ошибок, провижининг Grafana. `sd35-lora.sh` и `worker-reload.sh` ждут строк журнала, которых после перехода на загрузку по требованию нет. Часть скриптов оставляет свои переменные в `deploy/.env`.
- **`scripts/diagnose/<model>.sh`**: одна модель картинок. Меняет `COMPOSE_PROFILES` и флаги квантизации и выгрузки в `deploy/.env` без отката, пересобирает и поднимает воркер и шлюз, делает один запрос. Общие функции в `_lib.sh`, он ждёт строку журнала `Worker reachable: <id>`.
- **`scripts/benchmark.py`**: последовательные замеры задержки, `--endpoint chat|images|tts --n 10 --url http://localhost:8000`.
