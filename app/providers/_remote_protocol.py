from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx

ResponseKind = Literal["bytes", "json", "json_field"]


@dataclass(frozen=True)
class JsonEndpoint:
    """Worker endpoint that takes a JSON body keyed off `payload_key`."""
    path: str
    payload_key: str
    response_kind: ResponseKind = "json"
    response_field: str | None = None


@dataclass(frozen=True)
class MultipartEndpoint:
    """Worker endpoint that takes a multipart upload (one file + optional form fields)."""
    path: str
    file_field: str = "file"
    default_filename: str = "data.bin"
    content_type: str = "application/octet-stream"
    response_kind: ResponseKind = "json"
    response_field: str | None = None


def _unwrap(resp: httpx.Response, kind: ResponseKind, field: str | None) -> Any:
    if kind == "bytes":
        return resp.content
    payload = resp.json()
    if kind == "json_field":
        assert field is not None, "response_field required for json_field response_kind"
        return payload[field]
    return payload


async def call_json(
    client: httpx.AsyncClient,
    endpoint: JsonEndpoint,
    primary: Any,
    *,
    extra: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    send: Any = None,
) -> Any:
    """POST a JSON body and unwrap the response per `endpoint.response_kind`.

    `send` lets callers wrap the underlying request in retry/circuit-breaker logic.
    Default: direct client.post.
    """
    body = {endpoint.payload_key: primary, **(extra or {})}
    if send is None:
        resp = await client.post(endpoint.path, json=body, headers=headers)
    else:
        resp = await send(lambda: client.post(endpoint.path, json=body, headers=headers))
    resp.raise_for_status()
    return _unwrap(resp, endpoint.response_kind, endpoint.response_field)


async def call_multipart(
    client: httpx.AsyncClient,
    endpoint: MultipartEndpoint,
    file_bytes: bytes,
    *,
    filename: str | None = None,
    form: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """POST a multipart upload (single file) and unwrap the response.

    Multipart bodies are consumed once on send, so we never auto-retry these.
    """
    files = {
        endpoint.file_field: (
            filename or endpoint.default_filename,
            file_bytes,
            endpoint.content_type,
        )
    }
    resp = await client.post(endpoint.path, files=files, data=form, headers=headers)
    resp.raise_for_status()
    return _unwrap(resp, endpoint.response_kind, endpoint.response_field)
