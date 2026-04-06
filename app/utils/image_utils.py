from __future__ import annotations

import io


def pil_to_png_bytes(image) -> bytes:
    """Convert a PIL Image to PNG bytes."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
