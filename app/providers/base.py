from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.config import ModelConfig


class BaseProvider(ABC):
    """Lifecycle + metadata base for every model provider."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self.config.id

    @property
    def vram_mb(self) -> int:
        """GPU VRAM required by this model in megabytes (0 means CPU-only)."""
        return self.config.model.get("vram_mb", 0)

    @abstractmethod
    async def load(self, model_dir: str) -> None:
        """Load model weights into GPU/RAM."""

    @abstractmethod
    async def unload(self) -> None:
        """Release model resources and free GPU memory."""

    def is_loaded(self) -> bool:
        return self._loaded

    async def get_stats(self) -> dict:
        """Return a live resource-usage snapshot used by the memory watchdog."""
        return {
            "model": self.model_id,
            "loaded": self.is_loaded(),
            "declared_vram_mb": self.vram_mb,
            "vram_used_mb": self.vram_mb if self.is_loaded() else 0,
        }


class ImageProvider(BaseProvider):
    """Interface for text-to-image models."""

    @abstractmethod
    async def generate(self, prompt: str, **params: Any) -> bytes:
        """Generate an image from `prompt`. Returns PNG bytes."""


class TextProvider(BaseProvider):
    """Interface for text/chat generation models."""

    @abstractmethod
    async def generate(self, messages: list[dict], **params: Any) -> dict:
        """Run chat completion. Returns an OpenAI-format response dict."""

    async def generate_stream(self, messages: list[dict], **params: Any):
        """Stream chat completion as OpenAI SSE chunks. Optional — override to enable."""
        raise NotImplementedError("Streaming not supported by this provider")


class TtsProvider(BaseProvider):
    """Interface for text-to-speech models."""

    @abstractmethod
    async def synthesize(self, text: str, **params: Any) -> bytes:
        """Synthesize speech from `text`. Returns audio bytes."""


class SttProvider(BaseProvider):
    """Interface for speech-to-text / ASR models."""

    @abstractmethod
    async def transcribe(self, audio: bytes, **params: Any) -> dict:
        """Transcribe `audio` to `{"text": str, ...}` (optionally language/duration/segments)."""


class ImageUpscaleProvider(BaseProvider):
    """Interface for super-resolution / upscaling models."""

    @abstractmethod
    async def upscale(self, image: bytes, **params: Any) -> bytes:
        """Upscale `image` by the provider's configured factor. Returns PNG bytes."""
