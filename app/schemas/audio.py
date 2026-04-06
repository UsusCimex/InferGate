from __future__ import annotations

from pydantic import BaseModel


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = 1.0
