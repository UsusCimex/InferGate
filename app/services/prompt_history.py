from __future__ import annotations

import itertools
from collections import deque
from typing import Any


class PromptHistory:
    """The last image requests of this gateway instance for the web page; lost on restart."""

    def __init__(self, size: int) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=size)

    def record(self, entry: dict[str, Any]) -> None:
        self._entries.append(entry)

    def recent(self, limit: int) -> list[dict[str, Any]]:
        """Newest first."""
        return list(itertools.islice(reversed(self._entries), limit))

    def by_image(self) -> dict[str, dict[str, Any]]:
        """The latest request per cache key of its image."""
        return {entry["image"]: entry for entry in self._entries if entry.get("image")}
