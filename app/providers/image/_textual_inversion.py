from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class TextualInversionRegistry:
    """Dedup set of textual-inversion embeddings registered in a pipeline's tokenizer."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._loaded: set[tuple[str, str | None, Any]] = set()
        self._lock = threading.Lock()

    def apply(self, pipe: Any, tis: list[dict] | None, model_dir: str) -> None:
        """Register any not-yet-loaded TIs from `tis` into the pipeline tokenizer."""
        if not tis:
            return
        if pipe is None:
            return
        if not hasattr(pipe, "load_textual_inversion"):
            raise ValueError(
                f"Pipeline for {self._model_id} does not support textual inversions"
            )

        with self._lock:
            for spec in tis:
                repo_id = spec["id"]
                weight_file = spec.get("weight_file")
                token = spec.get("token")
                # Normalise list tokens to tuples — they'd be unhashable as set keys otherwise.
                token_key: Any = tuple(token) if isinstance(token, list) else token
                cache_key = (repo_id, weight_file, token_key)

                if cache_key in self._loaded:
                    logger.debug("TI cache hit: %s (token=%s)", repo_id, token)
                    continue

                load_kwargs: dict[str, Any] = {"cache_dir": model_dir}
                if weight_file:
                    load_kwargs["weight_name"] = weight_file
                if token:
                    load_kwargs["token"] = token

                try:
                    logger.info(
                        "Registering textual inversion %s%s%s into %s",
                        repo_id,
                        f" (file={weight_file})" if weight_file else "",
                        f" (token={token})" if token else "",
                        self._model_id,
                    )
                    try:
                        pipe.load_textual_inversion(repo_id, **load_kwargs)
                    except Exception as single_call_err:
                        # SDXL pivotal TIs (dual clip_l + clip_g tensors in one file)
                        # need explicit per-encoder loading; single-call path rejects them.
                        if (
                            "clip_l" in str(single_call_err)
                            and "clip_g" in str(single_call_err)
                            and weight_file
                            and hasattr(pipe, "text_encoder_2")
                            and hasattr(pipe, "tokenizer_2")
                        ):
                            logger.info(
                                "Falling back to per-encoder pivotal TI loading for %s", repo_id
                            )
                            self._load_pivotal(pipe, repo_id, weight_file, token, model_dir)
                        else:
                            raise
                except Exception as e:
                    raise ValueError(
                        f"Failed to load textual inversion '{repo_id}'"
                        f"{f' (file={weight_file})' if weight_file else ''}"
                        f"{f' (token={token})' if token else ''}: {e}"
                    ) from e

                self._loaded.add(cache_key)

    @staticmethod
    def _load_pivotal(
        pipe: Any,
        repo_id: str,
        weight_file: str,
        token: Any,
        model_dir: str,
    ) -> None:
        """Load an SDXL pivotal TI by splitting its clip_l/clip_g tensors across encoders."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        local_path = hf_hub_download(
            repo_id=repo_id, filename=weight_file, cache_dir=model_dir
        )
        sd = load_file(local_path)
        if "clip_l" not in sd or "clip_g" not in sd:
            raise ValueError(
                f"Pivotal TI fallback expected tensors 'clip_l' and 'clip_g' "
                f"in {weight_file}, got {list(sd.keys())}"
            )
        pipe.load_textual_inversion(
            sd["clip_l"],
            token=token,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
        )
        pipe.load_textual_inversion(
            sd["clip_g"],
            token=token,
            text_encoder=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer_2,
        )
