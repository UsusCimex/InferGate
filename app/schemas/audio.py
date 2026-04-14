from __future__ import annotations

from pydantic import BaseModel, Field


class AudioSpeechRequest(BaseModel):
    model: str | None = None
    input: str = Field(..., min_length=1, max_length=100000)
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
