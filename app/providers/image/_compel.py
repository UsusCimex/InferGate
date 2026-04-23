from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# A1111-style weight syntax: (word:1.5) / (word, phrase:0.8) / (word:-1.2)
# Simple regex — catches the common form without trying to parse the full
# compel grammar. If this matches, we route the prompt through compel;
# otherwise we pass the raw string to the pipeline (cheaper, no semantic shift).
_WEIGHT_RE = re.compile(r"\([^()]+:\s*[-+]?\d+\.?\d*\s*\)")


def has_weight_syntax(*prompts: str | None) -> bool:
    return any(p and _WEIGHT_RE.search(p) for p in prompts)


class CompelAdapter:
    """Wraps a compel.Compel instance and its detected pipeline mode.

    Modes: "sdxl" for dual-encoder pipelines (penultimate hidden states +
    pooled embedding on encoder #2), "sd15" for single-encoder pipelines.
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._compel: Any = None
        self._mode: str | None = None

    @property
    def available(self) -> bool:
        return self._compel is not None

    def init(self, pipeline: Any) -> None:
        """Detect pipeline layout and build a Compel instance. No-op on error."""
        try:
            from compel import Compel, ReturnedEmbeddingsType
        except ImportError:
            logger.info(
                "compel not installed; prompt weighting unavailable for %s",
                self._model_id,
            )
            return

        pipe_class = type(pipeline).__name__
        # SD3 / FLUX use T5-mixed embeds ([B,154,4096] / T5-only) — incompatible
        # with Compel's SDXL [B,77,2048] output. Detect by class name because
        # SD3 duck-types as SDXL via tokenizer_2 + text_encoder_2.
        if "StableDiffusion3" in pipe_class or pipe_class.startswith("Flux"):
            logger.info(
                "Compel skipped for %s (%s): architecture uses T5-mixed embeds "
                "that are incompatible with compel's SDXL output shape",
                self._model_id, pipe_class,
            )
            return

        try:
            if (
                hasattr(pipeline, "tokenizer_2")
                and hasattr(pipeline, "text_encoder_2")
                and pipeline.tokenizer_2 is not None
                and pipeline.text_encoder_2 is not None
            ):
                self._compel = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                )
                self._mode = "sdxl"
                logger.info("Compel initialised for %s (sdxl dual-encoder)", self._model_id)
            elif (
                hasattr(pipeline, "tokenizer")
                and hasattr(pipeline, "text_encoder")
                and pipeline.tokenizer is not None
                and pipeline.text_encoder is not None
            ):
                self._compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                self._mode = "sd15"
                logger.info("Compel initialised for %s (single-encoder)", self._model_id)
            else:
                logger.info(
                    "Compel skipped for %s (no compatible tokenizer/text_encoder)",
                    self._model_id,
                )
        except Exception as e:
            logger.warning(
                "Compel init failed for %s: %s — weighting disabled", self._model_id, e
            )
            self._compel = None
            self._mode = None

    def apply(self, prompt: str, negative_prompt: str | None, defaults: dict[str, Any]) -> None:
        """Replace `prompt`/`negative_prompt` in `defaults` with compel embeds.

        Removes any conflicting string-prompt keys so the pipeline's
        mutual-exclusion checks don't reject the call.
        """
        defaults.pop("negative_prompt", None)
        if self._mode == "sdxl":
            p_embeds, p_pooled = self._compel(prompt)
            defaults["prompt_embeds"] = p_embeds
            defaults["pooled_prompt_embeds"] = p_pooled
            if negative_prompt:
                n_embeds, n_pooled = self._compel(negative_prompt)
                defaults["negative_prompt_embeds"] = n_embeds
                defaults["negative_pooled_prompt_embeds"] = n_pooled
        else:  # sd15
            defaults["prompt_embeds"] = self._compel(prompt)
            if negative_prompt:
                defaults["negative_prompt_embeds"] = self._compel(negative_prompt)
