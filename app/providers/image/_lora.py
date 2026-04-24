from __future__ import annotations

import itertools
import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class LoraCache:
    """LRU cache of LoRA adapters registered in a diffusers pipeline."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._cache: OrderedDict[tuple[str, str | None], str] = OrderedDict()
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def apply(
        self,
        pipe: Any,
        loras: list[dict] | None,
        lora_cfg: dict[str, Any],
        model_dir: str,
    ) -> None:
        """Activate the requested LoRA set on `pipe`, loading new adapters as needed."""
        if pipe is None:
            return

        if not loras:
            if hasattr(pipe, "disable_lora"):
                pipe.disable_lora()
            return

        max_loaded = int(lora_cfg.get("max_loaded", 8))
        max_per_request = int(lora_cfg.get("max_per_request", 5))
        if len(loras) > max_per_request:
            raise ValueError(
                f"loras: {len(loras)} adapters requested, max {max_per_request}"
            )

        if not hasattr(pipe, "load_lora_weights"):
            raise ValueError(
                f"Pipeline for {self._model_id} does not support LoRA loading"
            )

        with self._lock:
            active_names: list[str] = []
            active_weights: list[float] = []

            for spec in loras:
                repo_id = spec["id"]
                weight_file = spec.get("weight_file")
                cache_key = (repo_id, weight_file)

                if cache_key in self._cache:
                    adapter_name = self._cache[cache_key]
                    self._cache.move_to_end(cache_key)
                    logger.debug("LoRA cache hit: %s → %s", repo_id, adapter_name)
                else:
                    adapter_name = spec.get("adapter_name") or f"lora_{next(self._counter)}"
                    while adapter_name in self._cache.values():
                        adapter_name = f"lora_{next(self._counter)}"

                    load_kwargs: dict[str, Any] = {
                        "cache_dir": model_dir,
                        "adapter_name": adapter_name,
                    }
                    if weight_file:
                        load_kwargs["weight_name"] = weight_file

                    try:
                        logger.info(
                            "Loading LoRA %s%s into %s as '%s'",
                            repo_id,
                            f" (file={weight_file})" if weight_file else "",
                            self._model_id,
                            adapter_name,
                        )
                        pipe.load_lora_weights(repo_id, **load_kwargs)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to load LoRA '{repo_id}'"
                            f"{f' (file={weight_file})' if weight_file else ''}: {e}"
                        ) from e

                    self._cache[cache_key] = adapter_name

                    while len(self._cache) > max_loaded:
                        evict_key, evict_name = self._cache.popitem(last=False)
                        logger.info(
                            "Evicting LoRA adapter '%s' (%s) — cache full (%d)",
                            evict_name, evict_key[0], max_loaded,
                        )
                        try:
                            pipe.delete_adapters([evict_name])
                        except Exception as e:
                            logger.warning("delete_adapters(%s) failed: %s", evict_name, e)

                active_names.append(adapter_name)
                active_weights.append(float(spec.get("weight", 1.0)))

            if hasattr(pipe, "enable_lora"):
                pipe.enable_lora()
            pipe.set_adapters(active_names, adapter_weights=active_weights)
            logger.debug(
                "Active LoRAs for request: %s",
                list(zip(active_names, active_weights, strict=True)),
            )
