"""YAML configuration file loaders."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from app.config.models import ModelConfig
from app.config.server import ServerConfig

logger = logging.getLogger(__name__)


def load_server_config(path: str | Path = "config/server.yaml") -> ServerConfig:
    path = Path(path)
    if not path.exists():
        return ServerConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ServerConfig(**data)


def load_single_model_config(path: str | Path) -> ModelConfig:
    """Load a single model config from a YAML file (used by workers)."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ModelConfig(**data)


def load_model_configs(models_dir: str | Path = "config/models") -> list[ModelConfig]:
    models_dir = Path(models_dir)
    configs: list[ModelConfig] = []
    if not models_dir.exists():
        return configs
    for yaml_file in sorted(models_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f) or {}
        try:
            config = ModelConfig(**data)
            configs.append(config)
        except Exception as e:
            logger.warning("Failed to load %s: %s", yaml_file, e)
    return configs
