"""The web page at /ui: the page itself, prompt history, gallery and cached images."""
from __future__ import annotations

import base64
import hashlib
import logging
import re

import pytest
from httpx import ASGITransport, AsyncClient

import app.main as main_module
from app.config import ServerConfig, load_server_config
from app.services.prompt_history import PromptHistory

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _inline_hash(html: str, tag: str) -> str:
    content = re.search(rf"<{tag}>(.*?)</{tag}>", html, re.DOTALL).group(1)
    return f"'sha256-{base64.b64encode(hashlib.sha256(content.encode()).digest()).decode()}'"


@pytest.mark.asyncio
async def test_page_allows_only_its_own_style_and_script(client):
    resp = await client.get("/ui")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    csp = resp.headers["content-security-policy"]
    assert f"script-src {_inline_hash(resp.text, 'script')};" in csp
    assert f"style-src {_inline_hash(resp.text, 'style')};" in csp
    assert "unsafe-inline" not in csp


@pytest.mark.asyncio
async def test_history_lists_image_requests_newest_first(client):
    await client.post("/v1/images/generations", json={"prompt": "a fox", "seed": 1, "size": "512x512"})
    await client.post("/v1/images/generations", json={"prompt": "a fox", "seed": 1, "size": "512x512"})
    await client.post("/v1/images/generations", json={"prompt": "a lake", "n": 2})

    history = (await client.get("/ui/history")).json()["data"]

    assert [(e["prompt"], e["cache"]) for e in history] == [
        ("a lake", "DISABLED"), ("a fox", "HIT"), ("a fox", "MISS"),
    ]
    lake, hit, miss = history
    assert lake["image"] is None and lake["n"] == 2
    assert hit["image"] == miss["image"]
    assert miss["model"] == "test-image" and miss["seed"] == 1 and miss["size"] == "512x512"
    assert miss["edit"] is False
    assert (await client.get("/ui/history", params={"limit": 1})).json()["data"] == [lake]


@pytest.mark.asyncio
async def test_history_marks_edits(client):
    await client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _PNG, "image/png")},
        data={"prompt": "add a hat", "model": "test-image"},
    )

    [entry] = (await client.get("/ui/history")).json()["data"]
    assert entry["prompt"] == "add a hat" and entry["edit"] is True


@pytest.mark.asyncio
async def test_gallery_shows_cached_images_with_their_request(client):
    await client.post("/v1/images/generations", json={"prompt": "a fox", "seed": 1})

    [item] = (await client.get("/ui/gallery")).json()["data"]
    assert item["model_id"] == "test-image"
    assert item["request"]["prompt"] == "a fox"

    image = await client.get(f"/ui/images/{item['key']}")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_gallery_and_image_route_skip_other_cached_media(client, services):
    key = "a" * 64
    await services["cache"].put(key, b"RIFF" + b"\x00" * 40, "test-tts", {"max_size_mb": 10})

    assert (await client.get("/ui/gallery")).json()["data"] == []
    assert (await client.get(f"/ui/images/{key}")).status_code == 404


@pytest.mark.asyncio
async def test_image_route_answers_404_for_unknown_and_malformed_keys(client):
    assert (await client.get(f"/ui/images/{'b' * 64}")).status_code == 404
    assert (await client.get("/ui/images/not-a-key")).status_code == 404


@pytest.mark.asyncio
async def test_no_history_is_kept_while_the_page_is_off(client):
    client._transport.app.state.prompt_history = None

    resp = await client.post("/v1/images/generations", json={"prompt": "a fox", "seed": 1})

    assert resp.status_code == 200
    assert (await client.get("/ui/history")).json()["data"] == []
    [item] = (await client.get("/ui/gallery")).json()["data"]
    assert item["request"] is None


def test_prompt_history_keeps_the_last_entries():
    history = PromptHistory(2)
    for i, key in enumerate(["k1", None, "k1"]):
        history.record({"prompt": f"p{i}", "image": key})

    assert [e["prompt"] for e in history.recent(10)] == ["p2", "p1"]
    assert history.by_image() == {"k1": {"prompt": "p2", "image": "k1"}}


def _app(monkeypatch, *, ui: bool, api_keys: list[str] | None = None):
    cfg = main_module.load_server_config()
    cfg.ui.enabled = ui
    if api_keys:
        cfg.auth.enabled = True
        cfg.auth.api_keys = api_keys
    monkeypatch.setattr(main_module, "load_server_config", lambda: cfg)
    return main_module.create_app()


@pytest.mark.asyncio
@pytest.mark.parametrize(("ui", "api_keys"), [(False, ["secret"]), (True, None)])
async def test_page_needs_ui_enabled_and_api_keys(monkeypatch, ui, api_keys):
    app = _app(monkeypatch, ui=ui, api_keys=api_keys)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/ui")).status_code == 404
        assert (await c.get("/ui/history", headers={"Authorization": "Bearer secret"})).status_code == 404


def test_page_without_auth_keeps_no_history_and_warns(caplog):
    cfg = ServerConfig()
    assert main_module._prompt_history(cfg) is None
    cfg.ui.enabled = True
    with caplog.at_level(logging.WARNING, logger="app.main"):
        assert main_module._prompt_history(cfg) is None
    assert "/ui needs auth.enabled with api_keys" in caplog.text
    cfg.auth.enabled, cfg.auth.api_keys = True, ["secret"]
    assert isinstance(main_module._prompt_history(cfg), PromptHistory)


@pytest.mark.asyncio
async def test_page_opens_without_a_key_and_its_data_needs_one(monkeypatch):
    app = _app(monkeypatch, ui=True, api_keys=["secret"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/ui")).status_code == 200
        assert (await c.get("/ui/history")).status_code == 401
        resp = await c.get("/ui/history", headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200
        assert resp.json() == {"data": []}


def test_server_config_reads_ui_enabled(monkeypatch):
    monkeypatch.delenv("UI_ENABLED", raising=False)
    assert load_server_config("config/server.yaml").ui.enabled is False
    monkeypatch.setenv("UI_ENABLED", "true")
    assert load_server_config("config/server.yaml").ui.enabled is True
