from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AudioSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str | None = None
    input: str = Field(..., min_length=1, max_length=100000)
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    # Honoured by models that advertise multilingual synthesis
    # (qwen3-tts-06b: 10 languages). Ignored silently by the rest.
    language: str | None = Field(None, max_length=32)


class TranscriptionSegment(BaseModel):
    """verbose_json segment — mirrors OpenAI's /v1/audio/transcriptions shape."""
    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    """Default (json) transcription response is `{text: ...}`; verbose_json
    adds optional language, duration, segments."""
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptionSegment] | None = None
