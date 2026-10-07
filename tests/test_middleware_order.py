"""The middleware stack of create_app: CORS and the request id wrap the API key check."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import app.main as main_module


@pytest.fixture
def secured_app(monkeypatch):
    cfg = main_module.load_server_config()
    cfg.auth.enabled = True
    cfg.auth.api_keys = ["secret"]
    cfg.cors.allow_origins = ["https://app.example"]
    monkeypatch.setattr(main_module, "load_server_config", lambda: cfg)
    app = main_module.create_app()

    @app.get("/v1/ping")
    async def ping():
        return {"ok": True}

    return app


@pytest.mark.asyncio
async def test_a_preflight_passes_without_a_key(secured_app):
    async with AsyncClient(transport=ASGITransport(app=secured_app), base_url="http://test") as client:
        resp = await client.options("/v1/ping", headers={
            "Origin": "https://app.example", "Access-Control-Request-Method": "GET",
        })

    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "https://app.example"


@pytest.mark.asyncio
async def test_a_request_without_a_key_is_refused_with_cors_and_request_id(secured_app):
    async with AsyncClient(transport=ASGITransport(app=secured_app), base_url="http://test") as client:
        resp = await client.get("/v1/ping", headers={"Origin": "https://app.example"})

    assert resp.status_code == 401
    assert resp.headers["access-control-allow-origin"] == "https://app.example"
    assert resp.headers.get("x-request-id")


@pytest.mark.asyncio
async def test_a_request_with_the_key_goes_through(secured_app):
    async with AsyncClient(transport=ASGITransport(app=secured_app), base_url="http://test") as client:
        resp = await client.get("/v1/ping", headers={"Authorization": "Bearer secret"})

    assert resp.status_code == 200
