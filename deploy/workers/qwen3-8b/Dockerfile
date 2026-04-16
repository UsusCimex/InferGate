ARG VLLM_IMAGE=vllm/vllm-openai:v0.19.0
FROM ${VLLM_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements/base.txt requirements/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir --system -r requirements/base.txt

COPY deploy/workers/qwen3-8b/requirements.txt requirements/worker.txt
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

ENTRYPOINT []
CMD ["uvicorn", "app.worker:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--no-access-log"]
