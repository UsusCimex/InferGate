from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("infergate.access")

# Paths to skip in access log (noisy healthchecks)
_SKIP_PATHS = {"/health", "/openapi.json"}


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Logs incoming requests with method, path, status, and duration.
    Suppresses noisy healthcheck logs.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        client = request.client.host if request.client else "?"
        logger.info(
            "%s %s %s -> %d (%d ms)",
            client,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
