from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_IMAGE_URL_CHARS = 64_000_000


class TextPart(BaseModel):
    """Text part of a multimodal message."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    """Inline image as a data URL."""
    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., max_length=MAX_IMAGE_URL_CHARS)
    detail: str | None = None


class ImageURLPart(BaseModel):
    """Image part of a multimodal message."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImageURLPart, Field(discriminator="type")]


class ChatMessage(BaseModel):
    """Chat request message: plain text or a list of text and image parts."""
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str | list[ContentPart]

    def image_urls(self) -> list[str]:
        if isinstance(self.content, str):
            return []
        return [part.image_url.url for part in self.content if isinstance(part, ImageURLPart)]


class AssistantMessage(BaseModel):
    """Chat response message."""
    role: str = "assistant"
    content: str


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "text"


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=500)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=131072)
    stream: bool = False
    response_format: ResponseFormat | None = None
    thinking: bool | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo
