from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from app.providers.base import TtsProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class KokoroTtsProvider(TtsProvider):
    """Kokoro TTS provider (lightweight, CPU-capable)."""

    def __init__(self, config):
        super().__init__(config)
        self._pipelines: dict[str, Any] = {}
        self._pipelines_lock = asyncio.Lock()
        default_voice = str(config.model.get("default_params", {}).get("voice", "af_heart"))
        self._default_lang = _lang_code(_resolve_voice(default_voice), "a")

    async def load(self, model_dir: str) -> None:
        logger.info("Loading %s from %s", self.model_id, self.config.model["hub_id"])
        await self._pipeline_for(self._default_lang)
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def _pipeline_for(self, lang_code: str) -> Any:
        """The G2P pipeline of `lang_code`; every pipeline shares the first one's model."""
        async with self._pipelines_lock:
            pipeline = self._pipelines.get(lang_code)
            if pipeline is None:
                import kokoro

                shared = next(iter(self._pipelines.values()), None)
                kwargs: dict[str, Any] = {
                    "lang_code": lang_code,
                    "repo_id": self.config.model["hub_id"],
                    "device": self.config.model.get("device"),
                }
                if shared is not None:
                    kwargs["model"] = shared.model
                loop = asyncio.get_running_loop()
                pipeline = await loop.run_in_executor(None, lambda: kokoro.KPipeline(**kwargs))
                self._pipelines[lang_code] = pipeline
            return pipeline

    async def unload(self) -> None:
        import gc

        self._pipelines.clear()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, gc.collect)
        try:
            import torch
            if torch.cuda.is_available():
                def _cuda_cleanup() -> None:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                await loop.run_in_executor(None, _cuda_cleanup)
        except ImportError:
            pass

        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def synthesize(self, text: str, **params: Any) -> bytes:
        import soundfile as sf

        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)

        voice = _resolve_voice(defaults.pop("voice", "af_heart"))
        speed = defaults.pop("speed", 1.0)
        output_format = defaults.pop("output_format", "mp3")
        pipeline = await self._pipeline_for(_lang_code(voice, self._default_lang))

        loop = asyncio.get_running_loop()
        samples_list = await loop.run_in_executor(
            None,
            lambda: list(pipeline(text, voice=voice, speed=speed)),
        )

        import numpy as np

        # Kokoro yields (graphemes, phonemes, audio) tuples.
        all_audio = np.concatenate([gs[2] for gs in samples_list])

        buf = io.BytesIO()
        sf.write(buf, all_audio, 24000, format=_sf_format(output_format))
        buf.seek(0)
        return buf.getvalue()


_VOICE_MAP = {
    "default": "af_heart",
    "alloy": "af_alloy",
    "nova": "af_nova",
    "shimmer": "af_bella",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
}


def _resolve_voice(voice: str) -> str:
    """Map OpenAI voice aliases onto Kokoro voice IDs."""
    return _VOICE_MAP.get(voice, voice)


_LANG_CODES = frozenset("abefhijpz")


def _lang_code(voice: str, fallback: str) -> str:
    """Kokoro voice ids open with their language: `bf_emma` is British English, `ef_dora` Spanish."""
    if len(voice) > 3 and voice[0] in _LANG_CODES and voice[1] in "fm" and voice[2] == "_":
        return voice[0]
    return fallback


def _sf_format(fmt: str) -> str:
    return {"mp3": "mp3", "wav": "wav", "flac": "flac", "opus": "ogg"}.get(fmt, "wav")
