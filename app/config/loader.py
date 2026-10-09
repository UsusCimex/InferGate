from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel

from app.config.models import ModelConfig
from app.config.server import ServerConfig

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file and resolve `${oc.env:...}` interpolations against the environment."""
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]


def _unknown_keys(model: type[BaseModel], data: dict[str, Any], prefix: str = "") -> list[str]:
    """Dotted paths of the keys in `data` that `model` has no field for."""
    unknown: list[str] = []
    for key, value in data.items():
        field = model.model_fields.get(key)
        if field is None:
            unknown.append(prefix + key)
            continue
        nested = field.annotation
        if isinstance(value, dict) and isinstance(nested, type) and issubclass(nested, BaseModel):
            unknown.extend(_unknown_keys(nested, value, f"{prefix}{key}."))
    return unknown


def load_server_config(path: str | Path = "config/server.yaml") -> ServerConfig:
    """Load `server.yaml`, returning defaults when the file is absent."""
    path = Path(path)
    if not path.exists():
        return ServerConfig()
    data = _load_yaml(path) or {}
    unknown = _unknown_keys(ServerConfig, data)
    if unknown:
        logger.warning("%s: unknown keys ignored: %s", path, ", ".join(unknown))
    return ServerConfig(**data)


def load_single_model_config(path: str | Path) -> ModelConfig:
    """Load one model YAML into a ModelConfig."""
    data = _load_yaml(Path(path)) or {}
    return ModelConfig(**data)


def load_model_configs(models_dir: str | Path = "config/models") -> list[ModelConfig]:
    """Load every model YAML under `models_dir`, skipping ones that fail to parse."""
    models_dir = Path(models_dir)
    configs: list[ModelConfig] = []
    if not models_dir.exists():
        return configs
    for yaml_file in sorted(models_dir.glob("*.yaml")):
        try:
            data = _load_yaml(yaml_file) or {}
            configs.append(ModelConfig(**data))
        except Exception as e:
            logger.warning("Failed to load %s: %s", yaml_file, e)
    return configs
