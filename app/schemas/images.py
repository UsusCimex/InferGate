from __future__ import annotations

from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    model: str | None = None
    prompt: str = Field(..., min_length=1, max_length=10000)
    n: int = Field(1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    seed: int | None = None


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
