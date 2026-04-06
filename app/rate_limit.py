from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter per client IP."""

    def __init__(self, app, requests_per_minute: int):
        super().__init__(app)
        self._rpm = requests_per_minute
        self._window = 60.0
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health/docs
        if request.url.path in ("/health", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Clean old entries
        timestamps = self._requests[client_ip]
        cutoff = now - self._window
        self._requests[client_ip] = [t for t in timestamps if t > cutoff]
        timestamps = self._requests[client_ip]

        if len(timestamps) >= self._rpm:
            retry_after = int(timestamps[0] - cutoff) + 1
            return JSONResponse(
                {
                    "error": {
                        "message": f"Rate limit exceeded: {self._rpm} requests per minute",
                        "type": "rate_limit_exceeded",
                    }
                },
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
        return await call_next(request)
