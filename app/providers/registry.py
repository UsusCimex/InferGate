from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

from app.providers.base import BaseProvider

if TYPE_CHECKING:
    pass

_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider(cls: type[BaseProvider]) -> type[BaseProvider]:
    """Decorator to register a provider class by name."""
    _REGISTRY[cls.__name__] = cls
    return cls


def get_provider_class(name: str) -> type[BaseProvider]:
    """Resolve provider class by name. Auto-discovers if not yet loaded."""
    if not _REGISTRY:
        _discover_providers()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown provider class '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def _discover_providers() -> None:
    """Auto-import all provider modules to trigger @register_provider."""
    providers_dir = Path(__file__).parent
    for subdir in ("image", "tts", "text"):
        pkg_path = providers_dir / subdir
        if not pkg_path.exists():
            continue
        package_name = f"app.providers.{subdir}"
        for info in pkgutil.iter_modules([str(pkg_path)]):
            importlib.import_module(f"{package_name}.{info.name}")
