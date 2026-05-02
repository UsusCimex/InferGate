from __future__ import annotations

from fastapi import UploadFile

_DEFAULT_CHUNK = 64 * 1024


class UploadTooLargeError(Exception):
    """Raised when a streamed upload exceeds the per-endpoint byte budget."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        super().__init__(f"upload exceeds {max_bytes} bytes")


async def read_with_limit(
    upload: UploadFile, max_bytes: int, *, chunk_size: int = _DEFAULT_CHUNK
) -> bytes:
    """Stream-read `upload` and raise UploadTooLargeError once `max_bytes` is exceeded.

    Reading in chunks bounds memory use to `chunk_size` even if the client sends a
    multi-GB body — the limit fires before the full payload is buffered.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")

    parts: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError(max_bytes)
        parts.append(chunk)
    return b"".join(parts)
