ARG GPU_BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${GPU_BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/base.txt requirements/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir --system -r requirements/base.txt

COPY deploy/workers/openaudio-s1-mini/requirements.txt requirements/worker.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir --system -r requirements/worker.txt

COPY app/ ./app/

RUN useradd -m -s /bin/bash infergate && \
    mkdir -p /app/models && chown -R infergate:infergate /app/models
USER infergate

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8001

CMD ["uvicorn", "app.worker:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--no-access-log"]
