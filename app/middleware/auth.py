from __future__ import annotations

import json

from starlette.types import ASGIApp, Receive, Scope, Send

_SKIP_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})


class ApiKeyMiddleware:
    """Pure ASGI middleware for API key authentication."""

    def __init__(self, app: ASGIApp, api_keys: list[str]) -> None:
        self.app = app
        self._api_keys = set(api_keys)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in _SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            token = auth_header

        if not token or token not in self._api_keys:
            body = json.dumps(
                {"error": {"message": "Invalid API key", "type": "authentication_error"}}
            ).encode()
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)
