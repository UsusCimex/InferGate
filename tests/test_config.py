"""Tests for configuration loading."""
from __future__ import annotations

import logging

import pytest

from app.config import (
    ModelConfig,
    ServerConfig,
    load_model_configs,
    load_server_config,
    load_single_model_config,
)


def test_default_server_config():
    config = load_server_config("nonexistent.yaml")
    assert isinstance(config, ServerConfig)
    assert config.queue.max_size == 50


def test_load_server_config(caplog):
    with caplog.at_level(logging.WARNING, logger="app.config.loader"):
        config = load_server_config("config/server.yaml")
    assert caplog.text == ""
    assert config.gpu.max_loaded_models == 3
    assert config.cache.enabled is True


def test_unknown_server_keys_are_reported(tmp_path, caplog):
    path = tmp_path / "server.yaml"
    path.write_text("port: 8000\ngpu:\n  device: cuda:0\n  max_loaded_models: 2\n")
    with caplog.at_level(logging.WARNING, logger="app.config.loader"):
        config = load_server_config(path)
    assert config.gpu.max_loaded_models == 2
    assert "unknown keys ignored: port, gpu.device" in caplog.text


def test_redis_cache_is_configurable(monkeypatch):
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("CACHE_REDIS_URL", "redis://redis:6379/1")
    cache = load_server_config("config/server.yaml").cache.model_dump()
    assert cache["backend"] == "redis"
    assert cache["redis_url"] == "redis://redis:6379/1"
    assert cache["redis_prefix"] == "infergate:cache"


def test_credentials_with_any_origin_refuse_to_start(monkeypatch):
    import app.main as main
    from app.config import CorsConfig

    monkeypatch.setattr(
        main, "load_server_config", lambda: ServerConfig(cors=CorsConfig(allow_credentials=True))
    )
    with pytest.raises(ValueError, match="allow_credentials"):
        main.create_app()


def test_defaults_name_stt_and_upscale_models():
    defaults = load_server_config("config/server.yaml").defaults.model_dump()
    assert defaults["stt"] == "whisper-base"
    assert defaults["upscale"] == "realesrgan-x4"


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
