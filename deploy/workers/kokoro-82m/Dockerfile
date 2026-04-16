FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/base.txt requirements/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir --system -r requirements/base.txt

COPY deploy/workers/kokoro-82m/requirements.txt requirements/worker.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir --system -r requirements/worker.txt && \
    python -m spacy download en_core_web_sm

COPY app/ ./app/

RUN useradd -m -s /bin/bash infergate && \
    mkdir -p /app/models && chown -R infergate:infergate /app/models
USER infergate

EXPOSE 8001

CMD ["uvicorn", "app.worker:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--no-access-log"]
