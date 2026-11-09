from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class JsonSchemaFormat(BaseModel):
    """OpenAI structured output: the answer matches `schema`."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., max_length=64)
    description: str | None = None
    schema_: dict[str, Any] | None = Field(None, alias="schema")
    strict: bool | None = None


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JsonSchemaFormat | None = None

    @model_validator(mode="after")
    def _schema_with_its_type(self) -> ResponseFormat:
        if (self.type == "json_schema") != (self.json_schema is not None):
            raise ValueError("json_schema goes with type json_schema and only with it")
        return self


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
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
