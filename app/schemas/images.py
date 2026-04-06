from __future__ import annotations

from pydantic import BaseModel


class ImageGenerationRequest(BaseModel):
    model: str | None = None
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    seed: int | None = None


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
