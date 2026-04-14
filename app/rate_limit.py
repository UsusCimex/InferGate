from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


_MAX_TRACKED_IPS = 10_000
_SKIP_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter per client IP."""

    def __init__(self, app, requests_per_minute: int):
        super().__init__(app)
        self._rpm = requests_per_minute
        self._window = 60.0
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = 0.0

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        cutoff = now - self._window

        # Periodic cleanup of stale IPs (every 60s)
        if now - self._last_cleanup > 60.0:
            self._last_cleanup = now
            stale = [ip for ip, ts in self._requests.items() if not ts or ts[-1] < cutoff]
            for ip in stale:
                del self._requests[ip]
            # Hard cap: evict oldest IPs if too many tracked
            if len(self._requests) > _MAX_TRACKED_IPS:
                sorted_ips = sorted(
                    self._requests, key=lambda ip: self._requests[ip][-1] if self._requests[ip] else 0
                )
                for ip in sorted_ips[: len(self._requests) - _MAX_TRACKED_IPS]:
                    del self._requests[ip]

        # Clean old entries for this IP
        timestamps = self._requests[client_ip]
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
