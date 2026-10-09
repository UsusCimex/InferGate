# Конфигурация

Три уровня: `config/server.yaml` (шлюз), `config/models/*.yaml` (файл на модель) и `deploy/.env` (переменные окружения, которые подставляются в оба YAML). Под другое железо меняются только переменные, YAML в репозитории не правится.

## `config/server.yaml`

| Ключ | По умолчанию | Назначение |
|---|---|---|
| `log_level` | `info` | уровень журнала шлюза |
| `auth.enabled`, `auth.api_keys` | `false`, `[]` | проверка `Authorization: Bearer <key>`, работает только с непустым списком ключей; `/health`, `/v1/health` и документация FastAPI открыты всегда |
| `gpu.max_loaded_models` | 3 (`GPU_MAX_LOADED_MODELS`) | сколько GPU-моделей (с `vram_mb > 0`) держать загруженными |
| `gpu.max_vram_budget_mb`, `gpu.vram_headroom_mb` | 0, 0 (`GPU_MAX_VRAM_BUDGET_MB`, `GPU_VRAM_HEADROOM_MB`) | бюджет объявленной VRAM, 0 выключает его. Для 12 ГБ 10000, для 24 ГБ 22000, для 48 ГБ 44000 |
| `gpu.pinned_models` | `[]` | модели, которые никогда не выгружаются |
| `gpu.category_reservations` | `{}` | минимум загруженных моделей категории, который вытеснение старается не нарушать |
| `gpu.watchdog_interval_seconds` | 15 (`GPU_WATCHDOG_INTERVAL_SECONDS`) | период `MemoryWatchdog`, 0 выключает его |
| `gpu.watchdog_vram_threshold`, `gpu.watchdog_ram_threshold` | 0.92, 0.90 | порог аварийного вытеснения по живой VRAM и предупреждения по RAM |
| `gpu.worker_url_template` | не задан | шаблон адреса воркера `http://...{id}...` вместо переменных `WORKER_URL_*` |
| `queue.max_size` | 50 | предел запросов в работе и в очереди, сверх него 503 `queue_full` |
| `cache.enabled`, `directory`, `max_total_size_gb`, `cleanup_interval_minutes` | `true`, `./cache`, 10, 30 | дисковый кэш ответов |
| `cache.backend` | `local` | `local` (SQLite и файлы) или `redis` |
| `cors.*` | `*` | разрешённые источники, методы, заголовки |
| `models_dir` | `./models` | папка весов для локального режима и `scripts/download_models.py` |
| `defaults.image`, `text`, `tts` | `sdxl-base`, `qwen3.5-4b`, `kokoro-82m` | модель, если в запросе нет `model` |
| `defaults.embedding_text`, `_audio`, `_image`, `_video` | E5, CLAP, CLIP, CLIP4Clip | то же для эмбеддингов |
| `rate_limit.enabled`, `requests_per_minute` | `false`, 60 | скользящее окно на IP клиента |
| `upload_limits.*` | картинка 20, аудио 25, видео 200, апскейл 50 МБ | предел загружаемых файлов (413) |

Ключи `host`, `port`, `workers` и `gpu.device` в файле есть, но шлюз их не использует: адрес и порт задаёт команда uvicorn в `Dockerfile.gateway`.

## YAML модели

Пример, `config/models/flux2-klein-4b.yaml`:

```yaml
id: flux2-klein-4b
display_name: "FLUX.2 Klein 4B"
category: image
provider_class: DiffusersImageProvider
enabled: ${oc.decode:${oc.env:FLUX2_KLEIN_4B_ENABLED,true}}

model:
  hub_id: "black-forest-labs/FLUX.2-klein-4B"
  revision: "main"
  torch_dtype: ${oc.env:FLUX2_KLEIN_4B_TORCH_DTYPE,bfloat16}
  vram_mb: ${oc.decode:${oc.env:FLUX2_KLEIN_4B_VRAM_MB,6000}}
  cpu_offload: ${oc.decode:${oc.env:FLUX2_KLEIN_4B_CPU_OFFLOAD,false}}
  sequential_cpu_offload: ${oc.decode:${oc.env:FLUX2_KLEIN_4B_SEQUENTIAL_OFFLOAD,true}}
  warmup: ${oc.decode:${oc.env:FLUX2_KLEIN_4B_WARMUP,true}}
  quantization: ${oc.decode:${oc.env:FLUX2_KLEIN_4B_QUANTIZATION,null}}
  quantize_components: ["transformer", "text_encoder"]
  default_params:
    num_inference_steps: ${oc.decode:${oc.env:FLUX2_KLEIN_4B_STEPS,4}}
    guidance_scale: ${oc.decode:${oc.env:FLUX2_KLEIN_4B_CFG,0.0}}
    width: 1024
    height: 1024

cache:
  enabled: true
  strategy: seed_only        # always | seed_only | never
  ttl_hours: null
  max_size_mb: 2048

queue:
  priority: ${oc.env:FLUX2_KLEIN_4B_PRIORITY,low}
  timeout_seconds: ${oc.decode:${oc.env:FLUX2_KLEIN_4B_TIMEOUT,300}}
  max_concurrent: ${oc.decode:${oc.env:FLUX2_KLEIN_4B_MAX_CONCURRENT,1}}

metadata:
  license: Apache-2.0
  tags: ["fast", "flux2", "apache2", "open"]
```

Необязательные ключи верхнего уровня: `worker_url` (адрес воркера вместо `WORKER_URL_*`) и `capabilities`: `vision` (картинки во входе chat/completions), `voice_clone_only` (синтез только через `/audio/speech/voice-clone`), `voices` (список разрешённых голосов `/audio/speech`). Тег `voice-clone` в `metadata.tags` отмечает модели клонирования для клиентов, PictoLex строит по нему их список. `category`: `image`, `text`, `tts`, `stt`, `upscale`, `embedding-text`, `embedding-audio`, `embedding-multimodal` или `embedding-video`.

Ключи блока `model` у `DiffusersImageProvider`: `hub_id`, `torch_dtype`, `variant`, `revision`, `drop_t5`, `quantization` (`nf4` или `int4` через bitsandbytes), `quantize_components`, `cpu_offload`, `sequential_cpu_offload`, `page_text_encoders` (текстовые энкодеры на GPU только на время кодирования), `vae_tiling`, `vae_hub_id`, `compel`, `refiner_hub_id`, `refiner_variant`, `warmup`, `lora.{max_loaded,max_per_request}`, `default_params`.

Правки YAML подхватываются на ходу: `ConfigWatcher` раз в 2 с сверяет время изменения файлов и перезагружает модель. Загруженному удалённому воркеру шлюз передаёт новый конфиг через `/reload`; воркер без загруженной модели увидит правки только после перезапуска контейнера.

## Переменные окружения

`deploy/.env` (образец `deploy/.env.example`) читает Docker Compose. В YAML переменные попадают через резолверы OmegaConf: `${oc.env:VAR,default}` для строк и `${oc.decode:${oc.env:VAR,default}}` для чисел, булевых значений и `null`.

- Имя `<ID>_<FIELD>`: id в верхнем регистре, `-` и `.` заменены на `_` (`QWEN3_5_4B_*` у `qwen3.5-4b`, `QWEN_IMAGE_*` у `qwen-image`). Исключения, эмбеддеры: `E5_BASE_*`, `CLAP_*`, `CLIP_*`, `CLIP4CLIP_*`, `SIGLIP_*`.
- В `.env.example` главные переменные, полный список в самих YAML.
- После правки `.env` воркер пересоздаётся: `docker compose ... up -d --force-recreate worker-<id>`.

| Группа | Переменные |
|---|---|
| Общие | `HF_TOKEN` (gated-модели), `GPU_BASE_IMAGE`, `VLLM_IMAGE`, `PORT`, `MODELS_DIR` (папка весов на хосте), `COMPOSE_PROFILES` |
| Пределы контейнеров | `GPU_WORKER_MEM_LIMIT` (16g), `CPU_WORKER_MEM_LIMIT` (6g), `GATEWAY_MEM_LIMIT` (2g), `GPU_WORKER_SHM_SIZE`, `PYTORCH_CUDA_ALLOC_CONF` |
| VRAM и сторож | `GPU_MAX_VRAM_BUDGET_MB`, `GPU_VRAM_HEADROOM_MB`, `GPU_MAX_LOADED_MODELS`, `GPU_WATCHDOG_*` |
| Модели | `<ID>_ENABLED`, `<ID>_QUANTIZATION`, `<ID>_CPU_OFFLOAD`, `<ID>_SEQUENTIAL_OFFLOAD`, `<ID>_STEPS`, `<ID>_CFG`, `<ID>_VRAM_MB`, `<ID>_MAX_CONCURRENT`, `<ID>_GPU_MEM_UTIL`, `<ID>_CONTEXT_LENGTH`, голоса и форматы TTS и др. |
| Связь шлюза с воркерами | `GATEWAY_REMOTE_CONNECT_TIMEOUT` (5 с), `GATEWAY_REMOTE_LOAD_TIMEOUT` (1800 с), `GATEWAY_REMOTE_GENERATE_TIMEOUT` (300 с), `GATEWAY_REMOTE_QUICK_TIMEOUT` (5 с), `GATEWAY_REMOTE_RETRY_ATTEMPTS`, `GATEWAY_REMOTE_RETRY_BACKOFF`, пул соединений |

Выгрузка на CPU: у `flux1-dev`, `flux1-schnell`, `flux2-klein-4b`, `qwen-image` и `sd35-medium` по умолчанию включена послойная выгрузка (`sequential_cpu_offload`), и она проверяется раньше обычной. Чтобы выгрузку отключить, нужны и `<ID>_SEQUENTIAL_OFFLOAD=false`, и `<ID>_CPU_OFFLOAD=false`. На 12 ГБ послойная выгрузка стоит минут на картинку, поэтому FLUX.2 klein, FLUX.1-dev и Z-Image там работают в nf4 без неё (подсказки в `.env.example`).

`GATEWAY_REMOTE_GENERATE_TIMEOUT` задаёт наименьшее ожидание ответа воркера: шлюз ждёт не меньше `queue.timeout_seconds` модели, чтобы первым срабатывал тайм-аут очереди. Не ответивший вовремя воркер даёт клиенту 504.

## Защита памяти

1. **Пределы контейнеров** (`*_MEM_LIMIT`): при переполнении OOM killer убивает контейнер, а не хост.
2. **Бюджет VRAM** (`gpu.max_vram_budget_mb` и `vram_headroom_mb`): перед загрузкой `ProviderManager` выгружает давно не использованные модели, пока сумма объявленных `vram_mb` не уложится в бюджет; если выгружать нечего, ответ 503 `insufficient_resources` вместо CUDA OOM. Закреплённые модели и модели с запросами в работе не выгружаются; если мешают только вторые, загрузка ждёт конца их запросов до `queue.timeout_seconds` загружаемой модели и только потом отвечает 503. Модель с `device: cpu` бюджет не занимает (`vram_mb` считается нулём) и не выгружается. `device` шлюз читает из YAML со своими переменными окружения, поэтому `<ID>_DEVICE=cpu` нужен и шлюзу, а не только воркеру.
3. **`MemoryWatchdog`**: раз в `watchdog_interval_seconds` спрашивает у воркеров живую VRAM (`/stats`: NVML, иначе torch) и при превышении порога выгружает самую давнюю GPU-модель, если таких загружено больше одной (vLLM заранее резервирует около 90% видеопамяти, так что одна модель порог превышает всегда). Модели на CPU в подсчёт не входят и не выгружаются. RAM хоста только пишется в журнал.

## Файлы переопределения Compose

Что не выражается переменными (несколько GPU, тома, секции `deploy`), задаётся файлом поверх базового:

```bash
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.server.yml up -d
```

Образец: `deploy/docker-compose.server.example.yml`, для TLS `deploy/docker-compose.tls.yml` ([deployment.md](deployment.md#https-через-caddy)). Переменные, заданные в таком файле только воркеру, шлюз не видит: планировщик и бюджет считаются по его собственным значениям.
