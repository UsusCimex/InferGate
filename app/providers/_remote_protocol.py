from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.providers.base import ImageFrame, WorkerStreamError

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


def image_frame_line(frame: ImageFrame) -> bytes:
    """One NDJSON line of a worker's image stream."""
    payload = {"final": frame.final, "b64_json": base64.b64encode(frame.png).decode()}
    return json.dumps(payload).encode() + b"\n"


def image_error_line(message: str, kind: str) -> bytes:
    """The last NDJSON line of an image stream that failed after its response started."""
    return json.dumps({"error": {"message": message, "type": kind}}).encode() + b"\n"


def parse_image_frame(line: str) -> ImageFrame:
    """Decode a line of a worker's image stream.

    An error line raises ValueError (invalid request) or RuntimeError, a malformed line
    WorkerStreamError.
    """
    try:
        payload = json.loads(line)
        error = payload.get("error")
        if error is None:
            png = base64.b64decode(payload["b64_json"], validate=True)
            return ImageFrame(png, final=bool(payload["final"]))
        invalid = error.get("type") == "invalid_request"
        message = error.get("message", "invalid request" if invalid else "worker error")
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise WorkerStreamError(f"malformed line in the worker's image stream: {e!r}") from e
    if invalid:
        raise ValueError(message)
    raise RuntimeError(message)
