from __future__ import annotations

import json
import time
from collections import defaultdict, deque

from starlette.types import ASGIApp, Receive, Scope, Send

_MAX_TRACKED_IPS = 10_000
_SKIP_PATHS = frozenset({"/health", "/v1/health", "/docs", "/redoc", "/openapi.json"})


class RateLimitMiddleware:
    """Sliding-window rate limiter, per client IP."""

    def __init__(self, app: ASGIApp, requests_per_minute: int) -> None:
        self.app = app
        self._rpm = requests_per_minute
        self._window = 60.0
        # Bursts can't exceed the per-minute budget, so the deque is capped at _rpm.
        self._requests: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max(1, self._rpm))
        )
        self._last_cleanup = 0.0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in _SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        now = time.monotonic()
        cutoff = now - self._window

        if now - self._last_cleanup > 60.0:
            self._last_cleanup = now
            stale = [ip for ip, ts in self._requests.items() if not ts or ts[-1] < cutoff]
            for ip in stale:
                del self._requests[ip]
            if len(self._requests) > _MAX_TRACKED_IPS:
                sorted_ips = sorted(
                    self._requests,
                    key=lambda ip: self._requests[ip][-1] if self._requests[ip] else 0,
                )
                for ip in sorted_ips[: len(self._requests) - _MAX_TRACKED_IPS]:
                    del self._requests[ip]

        timestamps = self._requests[client_ip]
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

        if len(timestamps) >= self._rpm:
            retry_after = str(int(timestamps[0] - cutoff) + 1)
            body = json.dumps({
                "error": {
                    "message": f"Rate limit exceeded: {self._rpm} requests per minute",
                    "type": "rate_limit_exceeded",
                }
            }).encode()
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                    [b"retry-after", retry_after.encode()],
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        timestamps.append(now)
        await self.app(scope, receive, send)
