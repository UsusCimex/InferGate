from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AudioSpeechRequest(BaseModel):
    """Request body for /v1/audio/speech."""
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str | None = None
    input: str = Field(..., min_length=1, max_length=100000)
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    language: str | None = Field(None, max_length=32)


class TranscriptionSegment(BaseModel):
    """One verbose_json segment from /v1/audio/transcriptions."""
    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    """Transcription response envelope (json + optional verbose_json fields)."""
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptionSegment] | None = None
