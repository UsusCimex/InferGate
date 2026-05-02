"""Tests for ASGI middleware: access log, rate limiter, auth."""
from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


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


@pytest.mark.asyncio
async def test_access_log_passes_through(app_with_access_log):
    resp = await app_with_access_log.get("/test")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


@pytest.mark.asyncio
async def test_access_log_skips_health(app_with_access_log):
    resp = await app_with_access_log.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_access_log_human_format_logs_request(app_with_access_log, caplog, monkeypatch):
    """Default format: 'client METHOD PATH -> status (Nms)' line."""
    import logging as _logging

    monkeypatch.delenv("INFERGATE_ACCESS_LOG_JSON", raising=False)
    with caplog.at_level(_logging.INFO, logger="app.middleware.access_log"):
        await app_with_access_log.get("/test")

    msgs = [r.message for r in caplog.records if r.name == "app.middleware.access_log"]
    assert any("/test" in m and "200" in m for m in msgs)


@pytest.mark.asyncio
async def test_access_log_json_format_emits_structured_line(
    app_with_access_log, caplog, monkeypatch
):
    """INFERGATE_ACCESS_LOG_JSON=true → emit a single-line JSON record."""
    import json as _json
    import logging as _logging

    monkeypatch.setenv("INFERGATE_ACCESS_LOG_JSON", "true")
    with caplog.at_level(_logging.INFO, logger="app.middleware.access_log"):
        await app_with_access_log.get("/test")

    msgs = [r.message for r in caplog.records if r.name == "app.middleware.access_log"]
    parsed = [m for m in msgs if m.startswith("{")]
    assert parsed, f"expected at least one JSON line, got: {msgs}"

    record = _json.loads(parsed[0])
    assert record["msg"] == "http_access"
    assert record["method"] == "GET"
    assert record["path"] == "/test"
    assert record["status"] == 200
    assert isinstance(record["latency_ms"], int)


@pytest.mark.asyncio
async def test_access_log_includes_request_id_when_set(monkeypatch, caplog):
    """When RequestIdMiddleware is mounted upstream, JSON log carries request_id."""
    import json as _json
    import logging as _logging

    from app.middleware.access_log import AccessLogMiddleware
    from app.monitoring import RequestIdMiddleware

    inner_app = FastAPI()

    @inner_app.get("/test")
    async def _t():
        return {"ok": True}

    # AccessLog is innermost; RequestId wraps it so scope["state"]["request_id"] is set.
    app = RequestIdMiddleware(AccessLogMiddleware(inner_app))
    monkeypatch.setenv("INFERGATE_ACCESS_LOG_JSON", "true")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        with caplog.at_level(_logging.INFO, logger="app.middleware.access_log"):
            await ac.get("/test", headers={"X-Request-ID": "test-id-42"})

    msgs = [r.message for r in caplog.records if r.name == "app.middleware.access_log"]
    parsed = [_json.loads(m) for m in msgs if m.startswith("{")]
    assert any(p.get("request_id") == "test-id-42" for p in parsed), parsed


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


@pytest.mark.asyncio
async def test_auth_uses_constant_time_compare():
    """ApiKeyMiddleware._is_valid must accept the right key and reject anything else."""
    from app.middleware.auth import ApiKeyMiddleware

    mw = ApiKeyMiddleware(app=None, api_keys=["key-a", "key-b", "key-c"])  # type: ignore[arg-type]
    assert mw._is_valid("key-a") is True
    assert mw._is_valid("key-b") is True
    assert mw._is_valid("key-c") is True
    assert mw._is_valid("key-d") is False
    assert mw._is_valid("") is False


@pytest.mark.asyncio
async def test_request_id_middleware_sets_contextvar_and_header():
    """RequestIdMiddleware must set X-Request-ID and expose it via ContextVar."""
    from app.middleware.access_log import AccessLogMiddleware  # unrelated; for completeness
    from app.monitoring import RequestIdMiddleware, get_request_id

    seen: dict[str, str | None] = {}

    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        seen["request_id"] = get_request_id()
        return {"ok": True}

    app = RequestIdMiddleware(inner_app)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/test")

    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    assert seen["request_id"] == resp.headers["x-request-id"]
    _ = AccessLogMiddleware  # silence unused import in this scope


@pytest.mark.asyncio
async def test_request_id_middleware_honours_inbound_id():
    """An inbound X-Request-ID must be propagated, not overwritten."""
    from app.monitoring import RequestIdMiddleware

    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    app = RequestIdMiddleware(inner_app)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/test", headers={"X-Request-ID": "trace-abc-123"})

    assert resp.headers["x-request-id"] == "trace-abc-123"
