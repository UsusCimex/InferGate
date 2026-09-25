from __future__ import annotations

import base64
import binascii
import io
import re
from typing import Any

_DATA_URL = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,(?P<data>.+)$", re.DOTALL)


def image_urls(messages: list[dict[str, Any]]) -> list[str]:
    """URLs of all image parts, in message order."""
    return [
        part["image_url"]["url"]
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]


def decode_data_url(url: str):
    """RGB PIL image from a base64 data:image URL; raises ValueError for anything else."""
    from PIL import Image, UnidentifiedImageError

    match = _DATA_URL.match(url)
    if not match:
        raise ValueError("only base64 data:image URLs are accepted")
    try:
        raw = base64.b64decode(match["data"], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError) as e:
        raise ValueError(f"invalid image data: {e}") from e


def text_of(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if part.get("type") == "text")


def with_system_instruction(messages: list[dict[str, Any]], instruction: str) -> list[dict[str, Any]]:
    """Messages with the instruction appended to the system message, or prepended as one."""
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        # The Qwen templates reject images in the system message, so it stays plain text.
        messages[0] = {**messages[0], "content": text_of(messages[0]["content"]) + "\n" + instruction}
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return messages
