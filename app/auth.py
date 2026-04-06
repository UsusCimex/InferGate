from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key authentication middleware."""

    def __init__(self, app, api_keys: list[str]):
        super().__init__(app)
        self._api_keys = set(api_keys)

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health check
        if request.url.path in ("/health", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            token = auth_header

        if token not in self._api_keys:
            return JSONResponse(
                {"error": {"message": "Invalid API key", "type": "authentication_error"}},
                status_code=401,
            )

        return await call_next(request)
