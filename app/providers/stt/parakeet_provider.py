from __future__ import annotations

import asyncio
import gc
import io
import logging
from typing import Any

from app.providers.base import SttProvider
from app.providers.registry import register_provider

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def decode_audio(audio: bytes):
    """Any container PyAV reads (WAV, MP3, AAC in MP4, Ogg) as 16 kHz mono float32 samples."""
    import av
    import numpy as np

    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(io.BytesIO(audio)) as container:
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


@register_provider
class ParakeetProvider(SttProvider):
    """NVIDIA Parakeet TDT through onnx-asr, from the ONNX export named by `model.hub_id`."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None

    def allow_patterns(self) -> list[str]:
        """The files of the export onnx-asr needs; the trailing `*` takes the external weights of a large encoder."""
        quantization = self.config.model.get("quantization") or ""
        suffix = f".{quantization}" if quantization else ""
        return ["config.json", "vocab.txt", f"encoder-model{suffix}.onnx*", f"decoder_joint-model{suffix}.onnx*"]

    async def load(self, model_dir: str) -> None:
        import onnx_asr
        from huggingface_hub import snapshot_download

        hub_id = self.config.model["hub_id"]
        quantization = self.config.model.get("quantization") or None
        threads = int(self.config.model.get("cpu_threads", 0))
        logger.info("Loading %s from %s (quantization=%s)", self.model_id, hub_id, quantization)

        def _load():
            import onnxruntime as rt

            path = snapshot_download(hub_id, cache_dir=model_dir, allow_patterns=self.allow_patterns())
            options = rt.SessionOptions()
            if threads:
                options.intra_op_num_threads = threads
            return onnx_asr.load_model(
                self.config.model["asr_model"], path, quantization=quantization,
                sess_options=options, providers=["CPUExecutionProvider"],
            )

        self._model = await asyncio.get_running_loop().run_in_executor(None, _load)
        self._loaded = True
        logger.info("Loaded %s", self.model_id)

    async def unload(self) -> None:
        self._model = None
        await asyncio.get_running_loop().run_in_executor(None, gc.collect)
        self._loaded = False
        logger.info("Unloaded %s", self.model_id)

    async def transcribe(self, audio: bytes, **params: Any) -> dict:
        defaults = dict(self.config.model.get("default_params", {}))
        defaults.update(params)
        response_format = str(defaults.get("response_format", "json"))

        def _run():
            waveform = decode_audio(audio)
            text = self._model.recognize(waveform, sample_rate=SAMPLE_RATE).strip() if len(waveform) else ""  # type: ignore[union-attr]
            return text, len(waveform) / SAMPLE_RATE

        try:
            text, duration = await asyncio.get_running_loop().run_in_executor(None, _run)
        except Exception as e:
            if type(e).__module__.startswith("av"):
                raise ValueError(f"cannot read the audio: {e}") from e
            raise
        result: dict[str, Any] = {"text": text}
        if response_format == "verbose_json":
            result["language"] = defaults.get("language") or None
            result["duration"] = duration
            result["segments"] = [{"id": 0, "start": 0.0, "end": duration, "text": text}] if text else []
        return result
