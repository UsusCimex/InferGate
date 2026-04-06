from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.config import ModelConfig


class BaseProvider(ABC):
    """Base provider. All providers inherit from this."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self.config.id

    @property
    def vram_mb(self) -> int:
        return self.config.model.get("vram_mb", 0)

    @abstractmethod
    async def load(self, model_dir: str) -> None:
        """Load model into GPU/RAM."""

    @abstractmethod
    async def unload(self) -> None:
        """Unload model, free memory."""

    def is_loaded(self) -> bool:
        return self._loaded


class ImageProvider(BaseProvider):
    """Interface for image generation."""

    @abstractmethod
    async def generate(self, prompt: str, **params: Any) -> bytes:
        """Generate image, return PNG bytes."""


class TextProvider(BaseProvider):
    """Interface for text generation."""

    @abstractmethod
    async def generate(self, messages: list[dict], **params: Any) -> dict:
        """Run chat completion, return OpenAI-format response dict."""

    async def generate_stream(self, messages: list[dict], **params: Any):
        """Stream chat completion. Yields OpenAI SSE chunks."""
        raise NotImplementedError("Streaming not supported by this provider")


class TtsProvider(BaseProvider):
    """Interface for text-to-speech."""

    @abstractmethod
    async def synthesize(self, text: str, **params: Any) -> bytes:
        """Synthesize speech, return audio bytes."""
