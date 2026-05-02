from __future__ import annotations

import hmac
import json

from starlette.types import ASGIApp, Receive, Scope, Send

_SKIP_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})


class ApiKeyMiddleware:
    """API-key auth: 401 when the Bearer token isn't in the configured allowlist."""

    def __init__(self, app: ASGIApp, api_keys: list[str]) -> None:
        self.app = app
        # Keep a list (not a set) so each comparison runs against every entry
        # at the same speed regardless of which key is valid.
        self._api_keys = list(api_keys)

    def _is_valid(self, token: str) -> bool:
        if not token:
            return False
        # OR over constant-time compare — never short-circuits on first match.
        valid = False
        for key in self._api_keys:
            if hmac.compare_digest(token, key):
                valid = True
        return valid

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in _SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()

        token = auth_header[7:] if auth_header.startswith("Bearer ") else auth_header

        if not self._is_valid(token):
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
