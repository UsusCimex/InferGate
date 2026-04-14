"""Tests for the standalone worker app."""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import ModelConfig, ModelCacheConfig, ModelQueueConfig, ModelMetadata
from tests.conftest import FakeTextProvider, FakeImageProvider, FakeTtsProvider


def _make_config(category: str, provider_class: str) -> ModelConfig:
    return ModelConfig(
        id=f"test-{category}",
        display_name=f"Test {category}",
        category=category,
        provider_class=provider_class,
        enabled=True,
        model={"hub_id": "test/test"},
        cache=ModelCacheConfig(),
        queue=ModelQueueConfig(),
        metadata=ModelMetadata(),
    )


@pytest_asyncio.fixture
async def text_worker():
    """Worker app serving a fake text model."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    config = _make_config("text", "FakeTextProvider")
    provider = FakeTextProvider(config)
    await provider.load(".")

    app.state.provider = provider
    app.state.config = config

    # Import worker routes
    from app.worker import health, load, unload, generate, synthesize
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/load", load, methods=["POST"])
    app.add_api_route("/unload", unload, methods=["POST"])
    app.add_api_route("/generate", generate, methods=["POST"])
    app.add_api_route("/synthesize", synthesize, methods=["POST"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://worker") as ac:
        yield ac


@pytest_asyncio.fixture
async def image_worker():
    """Worker app serving a fake image model."""
    from fastapi import FastAPI
    app = FastAPI()

    config = _make_config("image", "FakeImageProvider")
    provider = FakeImageProvider(config)
    await provider.load(".")

    app.state.provider = provider
    app.state.config = config

    from app.worker import health, load, generate
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/load", load, methods=["POST"])
    app.add_api_route("/generate", generate, methods=["POST"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://worker") as ac:
        yield ac


@pytest.mark.asyncio
async def test_worker_health(text_worker):
    resp = await text_worker.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "test-text"
    assert data["category"] == "text"


@pytest.mark.asyncio
async def test_worker_load(text_worker):
    resp = await text_worker.post("/load")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_worker_text_generate(text_worker):
    resp = await text_worker.post("/generate", json={
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_worker_image_generate(image_worker):
    resp = await image_worker.post("/generate", json={
        "prompt": "A red circle",
        "size": "512x512",
    })
    assert resp.status_code == 200
    # Should return PNG bytes
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
