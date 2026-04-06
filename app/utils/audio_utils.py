from __future__ import annotations


def get_audio_content_type(fmt: str) -> str:
    """Return the MIME type for an audio format string."""
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "opus": "audio/ogg",
        "ogg": "audio/ogg",
    }.get(fmt, "application/octet-stream")
