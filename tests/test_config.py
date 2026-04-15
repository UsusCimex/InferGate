"""Tests for configuration loading."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    ServerConfig,
    ModelConfig,
    load_server_config,
    load_model_configs,
    load_single_model_config,
)


def test_default_server_config():
    config = load_server_config("nonexistent.yaml")
    assert isinstance(config, ServerConfig)
    assert config.host == "0.0.0.0"
    assert config.port == 8000


def test_load_server_config():
    config = load_server_config("config/server.yaml")
    assert config.host == "0.0.0.0"
    assert config.gpu.max_loaded_models == 3
    assert config.cache.enabled is True


def test_load_model_configs():
    configs = load_model_configs("config/models")
    assert len(configs) > 0
    assert all(isinstance(c, ModelConfig) for c in configs)


def test_load_model_configs_empty_dir(tmp_path):
    configs = load_model_configs(str(tmp_path))
    assert configs == []


def test_load_model_configs_nonexistent():
    configs = load_model_configs("nonexistent_dir")
    assert configs == []


def test_load_model_configs_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: 123\n")  # id should be str but 123 is also valid; need missing required
    # Actually ModelConfig requires category, provider_class etc.
    configs = load_model_configs(str(tmp_path))
    assert len(configs) == 0  # should skip invalid


def test_load_single_model_config(tmp_path):
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text("""
id: test-model
display_name: Test
category: text
provider_class: VllmTextProvider
enabled: true
model:
  hub_id: test/test
""")
    config = load_single_model_config(str(yaml_path))
    assert config.id == "test-model"
    assert config.category == "text"


def test_model_config_worker_url():
    config = ModelConfig(
        id="remote",
        display_name="Remote",
        category="text",
        provider_class="VllmTextProvider",
        worker_url="http://worker:8001",
    )
    assert config.worker_url == "http://worker:8001"


def test_model_config_no_worker_url():
    config = ModelConfig(
        id="local",
        display_name="Local",
        category="text",
        provider_class="VllmTextProvider",
    )
    assert config.worker_url is None
