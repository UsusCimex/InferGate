# Модели

Модель: один YAML в `config/models/` и один контейнер-воркер в `deploy/docker-compose.yml`. Сейчас их 27 в девяти категориях; 17 включены по умолчанию (`enabled`), но контейнер стартует, только если выбран его профиль Compose.

## Каталог

`VRAM`: объявленный в YAML `vram_mb` по умолчанию, по нему шлюз планирует вытеснение; это не замер. Квантизация и выгрузка по умолчанию рассчитаны на видеокарту 12 ГБ и меняются переменными `deploy/.env` ([configuration.md](configuration.md#переменные-окружения)).

**Картинки** (`category: image`)

| id | Модель | Провайдер | VRAM, МБ | По умолчанию | Лицензия |
|---|---|---|---|---|---|
| `sdxl-base` | Stable Diffusion XL Base 1.0 | `DiffusersImageProvider` | 7000 | fp16, 896x896, VAE fp16-fix, compel и LoRA; Refiner через `SDXL_BASE_REFINER_HUB_ID` | OpenRAIL++ |
| `sd35-medium` | Stable Diffusion 3.5 Medium | `DiffusersImageProvider` | 5000 | без T5 (`drop_t5`), послойная выгрузка, LoRA | Stability AI Community |
| `flux1-schnell` | FLUX.1 Schnell | `DiffusersImageProvider` | 11000 | bf16, послойная выгрузка, 4 шага | Apache-2.0 |
| `flux1-dev` | FLUX.1 Dev (gated) | `DiffusersImageProvider` | 11000 | bf16, послойная выгрузка, 28 шагов | FLUX.1-dev Non-Commercial |
| `flux2-klein-4b` | FLUX.2 Klein 4B | `DiffusersImageProvider` | 6000 | bf16, послойная выгрузка; на 12 ГБ nf4 трансформера и текстового энкодера; референсы (`reference_images`) | Apache-2.0 |
| `qwen-image` | Qwen-Image | `DiffusersImageProvider` | 11000 | nf4, послойная выгрузка | Apache-2.0 |
| `hunyuan-dit` | Hunyuan-DiT v1.2 | `DiffusersImageProvider` | 7000 | fp16 | Tencent Hunyuan Community |
| `z-image-turbo` | Z-Image Turbo | `DiffusersImageProvider` | 12000 | bf16, 8 шагов; на 12 ГБ nf4 | Apache-2.0 |
| `janus-pro-1b` | Janus-Pro 1B (авторегрессия) | `JanusImageProvider` | 6000 | bf16, 384x384 | MIT |
| `janus-pro-7b` | Janus-Pro 7B (авторегрессия) | `JanusImageProvider` | 8000 | nf4, 384x384 | MIT |
| `meissonic` | Meissonic (маскированная, не авторегрессия) | `MeissonicImageProvider` | 10000 | fp16, 64 шага | Apache-2.0 |

Три архитектурных семейства работают за одним интерфейсом `ImageProvider`: диффузия и flow matching (SD, FLUX, Qwen-Image, Hunyuan-DiT, Z-Image), авторегрессия (Janus-Pro) и маскированное моделирование (Meissonic).

**Текст** (`category: text`, vLLM и llama.cpp)

| id | Модель | VRAM, МБ | По умолчанию | Лицензия |
|---|---|---|---|---|
| `qwen3.5-4b` | Qwen 3.5 4B (AWQ 4-bit) | 6000 | включена; контекст 8192, `thinking`, картинки во входе (`capabilities.vision`) | Apache-2.0 |
| `qwen3.5-9b` | Qwen 3.5 9B | 8000 | выключена | Apache-2.0 |
| `qwen3-8b` | Qwen 3 8B | 7000 | выключена | Apache-2.0 |
| `llama3.1-8b` | Llama 3.1 8B (gated) | 8000 | выключена | Llama 3.1 Community |
| `gemma-4-12b` | Gemma 4 12B (GGUF QAT Q4_0, llama.cpp) | 8192 | включена; контекст 8192, без размышлений; загрузка около 1 мин, около 60 токенов/с на RTX 5070 | Apache-2.0 |

**Озвучка** (`category: tts`)

| id | Модель | Провайдер | VRAM, МБ | Особенности | Лицензия |
|---|---|---|---|---|---|
| `kokoro-82m` | Kokoro 82M | `KokoroTtsProvider` | 0 (CPU) | английские голоса, американские `a*` и британские `b*`: язык по первой букве голоса | MIT |
| `voxcpm2` | VoxCPM2 | `VoxCpm2TtsProvider` | 8000 | четыре рассказчика (`vox_clara`, `vox_arthur`, `vox_lily`, `vox_daniel`) из `app/providers/tts/voxcpm2_voices/`, клонирование голоса, mp3 48 кГц, громкость -25 LUFS, seed 42, `torch.compile` | Apache-2.0 |
| `qwen3-tts-06b` | Qwen3-TTS 0.6B | `Qwen3TtsProvider` | 2500 | только клонирование (`voice_clone_only`), 10 языков, нужен `reference_text` | Apache-2.0 |
| `xtts-v2` | XTTS v2 (Coqui) | `XttsTtsProvider` | 2000 | выключена; только клонирование, 17 языков | CPML (некоммерческая) |
| `openaudio-s1-mini` | OpenAudio S1 Mini | `FishSpeechTtsProvider` | 4000 | выключена, экспериментальная | Apache-2.0 |

**Распознавание, апскейл, эмбеддинги**

| id | Категория | Провайдер | VRAM, МБ | Особенности | Лицензия |
|---|---|---|---|---|---|
| `whisper-base` | `stt` | `WhisperProvider` (faster-whisper) | 0 (CPU, int8) | `json`, `text`, `verbose_json`, `srt`, `vtt` | MIT |
| `parakeet-tdt-0.6b-v2` | `stt` | `ParakeetProvider` (onnx-asr) | 0 (CPU, int8) | только английский, с пунктуацией; `prompt`, `temperature` и `vad_filter` не нужны | CC-BY-4.0 |
| `parakeet-tdt-0.6b-v3` | `stt` | `ParakeetProvider` (onnx-asr) | 0 (CPU, int8) | 25 европейских языков, в том числе русский, язык определяет сама | CC-BY-4.0 |
| `realesrgan-x4` | `upscale` | `SpandrelUpscaleProvider` | 1500 | x4, вход до 2048 px по стороне, больше 1024 px идёт тайлами | BSD-3-Clause |
| `multilingual-e5-base` | `embedding-text` | `SentenceTransformerEmbeddingProvider` | 0 (CPU) | выключена | MIT |
| `clip-vit-base-patch32` | `embedding-multimodal` | `CLIPProvider` | 1500 | выключена; текст и картинки, 512 измерений | MIT |
| `siglip-base-multilingual` | `embedding-multimodal` | `SigLIPProvider` | 2000 | выключена; только свой профиль | Apache-2.0 |
| `clap-htsat-fused` | `embedding-audio` | `ClapEmbeddingProvider` | 4000 | выключена | CC-BY-4.0 |
| `clip4clip-webvid150k` | `embedding-video` | `CLIP4ClipProvider` | 2000 | выключена; 12 кадров на видео | MIT |

Модели по умолчанию (`defaults` в `config/server.yaml`): картинки `sdxl-base`, текст `qwen3.5-4b`, озвучка `kokoro-82m`, распознавание `whisper-base`, апскейл `realesrgan-x4`, эмбеддинги E5, CLAP, CLIP, CLIP4Clip. PictoLex сам указывает модели: `flux2-klein-4b`, `qwen3.5-4b`, `kokoro-82m`, `voxcpm2`.

## Профили Compose

Без `COMPOSE_PROFILES` (в `deploy/.env` или флагами `--profile`) поднимается только шлюз.

| Профиль | Воркеры |
|---|---|
| `text` | `qwen3.5-4b` |
| `image` | `sdxl-base` |
| `tts` | `kokoro-82m`, `voxcpm2` |
| `voice-clone` | `qwen3-tts-06b`, `xtts-v2` |
| `stt` | `whisper-base`, `parakeet-tdt-0.6b-v2`, `parakeet-tdt-0.6b-v3` |
| `upscale` | `realesrgan-x4` |
| `embedding` | E5, CLAP, CLIP, CLIP4Clip |
| `embedding-text`, `embedding-audio`, `embedding-image`, `embedding-video` | по одному эмбеддеру |
| `<id модели>` | воркер этой модели (например `flux2-klein-4b`, `siglip-base-multilingual`) |

Для PictoLex: `COMPOSE_PROFILES=flux2-klein-4b,qwen3.5-4b,kokoro-82m,voxcpm2`.

## Провайдеры

| `category` | `provider_class` | Что умеет |
|---|---|---|
| `image` | `DiffusersImageProvider` | любая модель diffusers: LoRA, Textual Inversion, веса compel (SDXL), смена планировщика, HighresFix, SDXL Refiner, img2img, inpaint, nf4 через bitsandbytes, превью шагов в потоке |
| `image` | `JanusImageProvider` | DeepSeek Janus-Pro (авторегрессия) |
| `image` | `MeissonicImageProvider` | Meissonic (пайплайн из репозитория авторов) |
| `text` | `VllmTextProvider` | любая LLM через vLLM: потоковый вывод, шаблоны чата, картинки во входе |
| `text` | `LlamaCppTextProvider` | GGUF через llama-server: воркер запускает его на `/load` и передаёт запросы чата; флаги сервера в `model.server_args`, образ `deploy/Dockerfile.llamacpp` |
| `tts` | `KokoroTtsProvider`, `VoxCpm2TtsProvider`, `Qwen3TtsProvider`, `XttsTtsProvider`, `FishSpeechTtsProvider` | синтез и клонирование голоса |
| `stt` | `WhisperProvider`, `ParakeetProvider` | faster-whisper (CTranslate2); Parakeet TDT из ONNX-экспорта через onnx-asr, звук любого формата читает PyAV |
| `upscale` | `SpandrelUpscaleProvider` | spandrel: Real-ESRGAN, SwinIR и др. |
| `embedding-*` | `SentenceTransformerEmbeddingProvider`, `CLIPProvider`, `SigLIPProvider`, `ClapEmbeddingProvider`, `CLIP4ClipProvider` | эмбеддинги текста, картинок, аудио, видео |

Базовые классы в `app/providers/base.py` (Image, Text, Tts, Stt, ImageUpscale и четыре вида эмбеддингов). Провайдер регистрируется декоратором `@register_provider` и лежит в одном из подпакетов `app/providers/{image,text,tts,stt,upscale,embedding}`.

## Как добавить модель

Модель, которой хватает существующего провайдера, добавляется без кода Python:

1. `config/models/<id>.yaml`: `id`, `display_name`, `category`, `provider_class`, `enabled`, блок `model` (`hub_id`, `vram_mb`, `gpu`, `torch_dtype`, квантизация, выгрузка, `default_params`), `cache`, `queue`, `metadata`, при необходимости `capabilities`. Значения, зависящие от железа, пишутся как `${oc.env:<ID>_<FIELD>,default}`, чтобы их можно было менять в `deploy/.env` ([configuration.md](configuration.md#yaml-модели)).
2. `deploy/workers/<id>/requirements.txt`: pip-зависимости воркера.
3. Строка в матрице `deploy/docker-bake.hcl`; для GGUF на llama.cpp - отдельная цель по образцу `worker-gemma-4-12b`.
4. Сервис `worker-<id>` в `deploy/docker-compose.yml` (по образцу соседнего) с профилем `<id>` и, если нужно, профилем категории.
5. Переменная у сервиса `gateway`: `WORKER_URL_<ID>=http://worker-<id>:8001`. В имени переменной id в верхнем регистре, всё, кроме букв и цифр, заменено на `_`; в имени сервиса точки заменены дефисами (`WORKER_URL_QWEN3_5_4B=http://worker-qwen3-5-4b:8001`). Без неё шлюз считает модель локальной и пытается загрузить её в своём процессе, где нет torch.

Для новой архитектуры пишется класс провайдера в подпакете своей категории с `@register_provider`, его имя указывается в `provider_class`.
