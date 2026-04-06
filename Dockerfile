# Базовый образ: PyTorch + CUDA + Python
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Быстрый пакетный менеджер
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Слой 1: ML-зависимости (vllm, diffusers, transformers)
COPY requirements/ml.txt requirements/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/ml.txt

# Слой 2: TTS-зависимости (kokoro)
COPY requirements/tts.txt requirements/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/tts.txt

# Слой 3: Базовые зависимости (fastapi, pydantic)
COPY requirements/base.txt requirements/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/base.txt

# Слой 4: Код приложения
COPY app/ ./app/
COPY scripts/ ./scripts/

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
