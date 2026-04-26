from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingRequest(BaseModel):
    """Request body for /v1/embeddings (OpenAI-compatible)."""
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str | None = None
    input: str | list[str] = Field(...)
    encoding_format: Literal["float"] = "float"


class EmbeddingItem(BaseModel):
    """One embedding vector inside an EmbeddingResponse."""
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Response envelope for /v1/embeddings (OpenAI-compatible)."""
    object: Literal["list"] = "list"
    model: str
    data: list[EmbeddingItem]
    usage: dict[str, int]


class AudioEmbeddingResponse(BaseModel):
    """Response envelope for /v1/embeddings/audio (multipart extension)."""
    model: str
    embedding: list[float]


class ImageEmbeddingResponse(BaseModel):
    """Response envelope for /v1/embeddings/image (multipart extension)."""
    model: str
    embedding: list[float]
