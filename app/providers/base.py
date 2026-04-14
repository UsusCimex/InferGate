"""Abstract base classes defining the provider interface for each modality.

All concrete providers (local or remote) inherit from one of these ABCs.
The gateway interacts with providers only through these interfaces.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.config import ModelConfig


class BaseProvider(ABC):
    """Base class for all model providers.

    Manages lifecycle (load/unload) and exposes model metadata.
    Subclasses must implement load() and unload() for resource management.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self.config.id

    @property
    def vram_mb(self) -> int:
        """GPU VRAM required by this model in megabytes. 0 means CPU-only."""
        return self.config.model.get("vram_mb", 0)

    @abstractmethod
    async def load(self, model_dir: str) -> None:
        """Load model weights into GPU/RAM. Called by ProviderManager."""

    @abstractmethod
    async def unload(self) -> None:
        """Release model resources and free GPU memory."""

    def is_loaded(self) -> bool:
        return self._loaded


class ImageProvider(BaseProvider):
    """Interface for image generation models (diffusers, etc.)."""

    @abstractmethod
    async def generate(self, prompt: str, **params: Any) -> bytes:
        """Generate image from text prompt. Returns PNG bytes."""


class TextProvider(BaseProvider):
    """Interface for text/chat generation models (vLLM, etc.)."""

    @abstractmethod
    async def generate(self, messages: list[dict], **params: Any) -> dict:
        """Run chat completion. Returns OpenAI-format response dict."""

    async def generate_stream(self, messages: list[dict], **params: Any):
        """Stream chat completion. Yields OpenAI SSE chunks.

        Optional — override in providers that support streaming.
        """
        raise NotImplementedError("Streaming not supported by this provider")


class TtsProvider(BaseProvider):
    """Interface for text-to-speech models (Kokoro, Fish Speech, etc.)."""

    @abstractmethod
    async def synthesize(self, text: str, **params: Any) -> bytes:
        """Synthesize speech from text. Returns audio bytes."""
