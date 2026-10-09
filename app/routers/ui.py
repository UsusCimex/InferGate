"""Web page at /ui and the JSON it reads besides /metrics and /v1/models."""
from __future__ import annotations

import base64
import functools
import hashlib
import re
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response

from app.dependencies import get_cache_manager, get_prompt_history

router = APIRouter()

_PAGE = Path(__file__).resolve().parent.parent / "static" / "ui.html"
_CACHE_KEY = re.compile(r"[0-9a-f]{64}")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@functools.cache
def _page() -> tuple[str, str]:
    """The page and a Content-Security-Policy that allows only its own inline style and script."""
    html = _PAGE.read_text(encoding="utf-8")
    csp = (
        "default-src 'none'; "
        f"script-src {_inline_hash(html, 'script')}; "
        f"style-src {_inline_hash(html, 'style')}; "
        "img-src blob: data:; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    return html, csp


def _inline_hash(html: str, tag: str) -> str:
    start = html.index(f"<{tag}>") + len(tag) + 2
    content = html[start:html.index(f"</{tag}>", start)]
    return f"'sha256-{base64.b64encode(hashlib.sha256(content.encode()).digest()).decode()}'"


@router.get("/ui", include_in_schema=False)
async def page():
    html, csp = _page()
    return HTMLResponse(html, headers={
        "Cache-Control": "no-cache",
        "Content-Security-Policy": csp,
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    })


@router.get("/ui/history")
async def history(limit: int = Query(50, ge=1, le=1000), prompts=Depends(get_prompt_history)):
    """Image requests served by this instance, newest first."""
    return {"data": prompts.recent(limit) if prompts else []}


@router.get("/ui/gallery")
async def gallery(
    limit: int = Query(16, ge=1, le=100),
    cache=Depends(get_cache_manager),
    prompts=Depends(get_prompt_history),
):
    """Newest cached images with the request that made them, when this instance remembers it."""
    requests = prompts.by_image() if prompts else {}
    images = await cache.recent_images(limit)
    for image in images:
        image["request"] = requests.get(image["key"])
    return {"data": images}


@router.get("/ui/images/{key}")
async def image(key: str, cache=Depends(get_cache_manager)):
    """A cached PNG; reading it counts no cache hit."""
    data = await cache.peek(key) if _CACHE_KEY.fullmatch(key) else None
    if data is None or not data.startswith(_PNG_SIGNATURE):
        return JSONResponse(
            {"error": {"message": "no cached image with this key", "type": "not_found"}},
            status_code=404,
        )
    return Response(data, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})
