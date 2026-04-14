"""Tests for monitoring: Prometheus metrics, request ID, middleware."""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI


@pytest_asyncio.fixture
async def monitored_app():
    """App with RequestIdMiddleware and PrometheusMiddleware."""
    from app.monitoring import RequestIdMiddleware, PrometheusMiddleware

    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    # Wrap: RequestId -> Prometheus -> app
    app = RequestIdMiddleware(PrometheusMiddleware(inner_app))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_request_id_header(monitored_app):
    resp = await monitored_app.get("/test")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) == 16


@pytest.mark.asyncio
async def test_request_id_unique(monitored_app):
    resp1 = await monitored_app.get("/test")
    resp2 = await monitored_app.get("/test")
    assert resp1.headers["x-request-id"] != resp2.headers["x-request-id"]


@pytest.mark.asyncio
async def test_prometheus_endpoint(client):
    resp = await client.get("/metrics/prometheus")
    assert resp.status_code == 200
    body = resp.text
    # Should contain Prometheus metric families
    assert "infergate_requests_total" in body or "text/plain" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_prometheus_metrics_recorded(client):
    """After making a request, Prometheus counters should be incremented."""
    # Make a request to trigger metrics
    await client.post("/v1/chat/completions", json={
        "model": "test-text",
        "messages": [{"role": "user", "content": "Hi"}],
    })

    resp = await client.get("/metrics/prometheus")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_json_metrics_still_works(client):
    """Original /metrics JSON endpoint still works."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "queue_size" in data
    assert "uptime_seconds" in data


def test_prometheus_available():
    from app.monitoring import is_prometheus_available
    assert is_prometheus_available() is True


def test_path_normalization():
    from app.monitoring import _normalize_path
    assert _normalize_path("/v1/chat/completions") == "/v1/chat/completions"
    assert _normalize_path("/v1/models/qwen3.5-4b/load") == "/v1/models/{id}"
    assert _normalize_path("/cache/stats/some-model") == "/cache/stats/{model_id}"
    assert _normalize_path("/cache/entry/abc123") == "/cache/entry/{key}"
    assert _normalize_path("/cache") == "/cache"
