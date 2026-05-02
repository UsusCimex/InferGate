from __future__ import annotations

from contextvars import ContextVar

# Set by RequestIdMiddleware on every inbound HTTP request.
# Read by remote providers to forward X-Request-ID into worker calls.
_REQUEST_ID: ContextVar[str | None] = ContextVar("infergate_request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    _REQUEST_ID.set(request_id)


def get_request_id() -> str | None:
    return _REQUEST_ID.get()
