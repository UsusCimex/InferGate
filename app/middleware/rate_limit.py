from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Protocol

from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import RateLimitBackend, RateLimitConfig

logger = logging.getLogger(__name__)

_MAX_TRACKED_IPS = 10_000
_SKIP_PATHS = frozenset({"/health", "/v1/health", "/docs", "/redoc", "/openapi.json"})
_WINDOW_S = 60.0
_REDIS_TIMEOUT_S = 1.0
# Seconds after a Redis failure when requests pass without waiting out its timeout.
_REDIS_RETRY_S = 5.0
_WARN_EVERY_S = 60.0

# Sliding window log: drop timestamps older than the window, admit when fewer than the limit remain.
# Returns {1, ""} when admitted, {0, oldest timestamp} when not (Lua numbers would be truncated to ints).
_SLIDING_WINDOW_LUA = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then
  return {0, redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')[2]}
end
redis.call('ZADD', KEYS[1], now, ARGV[4])
redis.call('PEXPIRE', KEYS[1], math.ceil(window * 1000))
return {1, ''}
"""


class RateLimiter(Protocol):
    requests_per_minute: int

    async def hit(self, client: str) -> int | None:
        """Count a request from `client`; None admits it, a number is the Retry-After in seconds."""
        ...

    async def close(self) -> None: ...


class MemoryRateLimiter:
    """Sliding window per client IP inside this process."""

    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        # Bursts can't exceed the per-minute budget, so the deque is capped at the limit.
        self._requests: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max(1, requests_per_minute))
        )
        self._last_cleanup = 0.0

    async def hit(self, client: str) -> int | None:
        now = time.monotonic()
        cutoff = now - _WINDOW_S

        if now - self._last_cleanup > _WINDOW_S:
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

        timestamps = self._requests[client]
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        if len(timestamps) >= self.requests_per_minute:
            return int(timestamps[0] - cutoff) + 1
        timestamps.append(now)
        return None

    async def close(self) -> None:
        return None


class RedisRateLimiter:
    """Sliding window per client IP in a Redis sorted set, shared by every gateway on that Redis.

    Without Redis the limiter admits requests, asks Redis again after a pause and logs a warning
    at most once a minute.
    """

    def __init__(
        self,
        requests_per_minute: int,
        url: str,
        prefix: str = "infergate:ratelimit",
        *,
        client: Any | None = None,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self._prefix = prefix
        self._owns_client = client is None
        if client is None:
            import redis.asyncio as redis_async

            client = redis_async.from_url(
                url, socket_timeout=_REDIS_TIMEOUT_S, socket_connect_timeout=_REDIS_TIMEOUT_S,
            )
        self._client = client
        self._script = client.register_script(_SLIDING_WINDOW_LUA)
        self._last_warning = 0.0
        self._down_until = 0.0

    async def hit(self, client: str) -> int | None:
        from redis.exceptions import RedisError

        if time.monotonic() < self._down_until:
            return None
        now = time.time()
        try:
            admitted, oldest = await self._script(
                keys=[f"{self._prefix}:{client}"],
                args=[now, _WINDOW_S, self.requests_per_minute, uuid.uuid4().hex],
            )
        except RedisError as e:
            self._down_until = time.monotonic() + _REDIS_RETRY_S
            if now - self._last_warning > _WARN_EVERY_S:
                self._last_warning = now
                logger.warning("Rate limit store unavailable, admitting requests: %s", e)
            return None
        if admitted:
            return None
        return max(1, int(float(oldest) + _WINDOW_S - now) + 1)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def make_rate_limiter(config: RateLimitConfig, default_redis_url: str) -> RateLimiter:
    """Limiter for `rate_limit` in server.yaml; Redis falls back to `default_redis_url` (the cache's)."""
    if config.backend == RateLimitBackend.REDIS:
        return RedisRateLimiter(
            config.requests_per_minute,
            config.redis_url or default_redis_url,
            config.redis_prefix,
        )
    return MemoryRateLimiter(config.requests_per_minute)


class RateLimitMiddleware:
    """Per client IP request limit; 429 with Retry-After once the minute's budget is spent."""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.app = app
        self._limiter = limiter or MemoryRateLimiter(requests_per_minute)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in _SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        retry_after = await self._limiter.hit(client[0] if client else "unknown")
        if retry_after is None:
            await self.app(scope, receive, send)
            return

        body = json.dumps({
            "error": {
                "message": f"Rate limit exceeded: {self._limiter.requests_per_minute} requests per minute",
                "type": "rate_limit_exceeded",
            }
        }).encode()
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
                [b"retry-after", str(retry_after).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})
