"""Tests for ASGI middleware: access log, rate limiter, auth."""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI
from fastapi.responses import JSONResponse


@pytest_asyncio.fixture
async def app_with_access_log():
    from app.middleware.access_log import AccessLogMiddleware

    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    @inner_app.get("/health")
    async def health():
        return {"status": "ok"}

    app = AccessLogMiddleware(inner_app)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def app_with_rate_limit():
    from app.middleware.rate_limit import RateLimitMiddleware

    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    @inner_app.get("/health")
    async def health():
        return {"status": "ok"}

    app = RateLimitMiddleware(inner_app, requests_per_minute=3)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def app_with_auth():
    from app.middleware.auth import ApiKeyMiddleware

    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    @inner_app.get("/health")
    async def health():
        return {"status": "ok"}

    app = ApiKeyMiddleware(inner_app, api_keys=["valid-key-123"])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --- Access Log Middleware ---

@pytest.mark.asyncio
async def test_access_log_passes_through(app_with_access_log):
    resp = await app_with_access_log.get("/test")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


@pytest.mark.asyncio
async def test_access_log_skips_health(app_with_access_log):
    resp = await app_with_access_log.get("/health")
    assert resp.status_code == 200


# --- Rate Limit Middleware ---

@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit(app_with_rate_limit):
    for _ in range(3):
        resp = await app_with_rate_limit.get("/test")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit(app_with_rate_limit):
    for _ in range(3):
        await app_with_rate_limit.get("/test")

    resp = await app_with_rate_limit.get("/test")
    assert resp.status_code == 429
    assert "rate_limit_exceeded" in resp.json()["error"]["type"]
    assert "retry-after" in resp.headers


@pytest.mark.asyncio
async def test_rate_limit_skips_health(app_with_rate_limit):
    # Health endpoint is exempt from rate limiting
    for _ in range(10):
        resp = await app_with_rate_limit.get("/health")
        assert resp.status_code == 200


# --- Auth Middleware ---

@pytest.mark.asyncio
async def test_auth_allows_valid_bearer(app_with_auth):
    resp = await app_with_auth.get("/test", headers={"Authorization": "Bearer valid-key-123"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_allows_raw_key(app_with_auth):
    resp = await app_with_auth.get("/test", headers={"Authorization": "valid-key-123"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_rejects_invalid_key(app_with_auth):
    resp = await app_with_auth.get("/test", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401
    assert "authentication_error" in resp.json()["error"]["type"]


@pytest.mark.asyncio
async def test_auth_rejects_missing_key(app_with_auth):
    resp = await app_with_auth.get("/test")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_rejects_empty_token(app_with_auth):
    resp = await app_with_auth.get("/test", headers={"Authorization": ""})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_skips_health(app_with_auth):
    resp = await app_with_auth.get("/health")
    assert resp.status_code == 200
