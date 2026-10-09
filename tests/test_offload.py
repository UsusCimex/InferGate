"""The offload mode of diffusion pipelines and the rejected two-flag settings."""
from __future__ import annotations

import pytest

from app.providers.image._offload import offload_mode


def test_offload_defaults_to_none():
    assert offload_mode("sdxl-base", {}) == "none"


@pytest.mark.parametrize("mode", ["none", "model", "sequential"])
def test_offload_mode_is_read(mode):
    assert offload_mode("flux2-klein-4b", {"offload": mode}) == mode


def test_unknown_offload_mode_fails():
    with pytest.raises(ValueError, match="offload must be one of"):
        offload_mode("flux2-klein-4b", {"offload": "true"})


def test_model_can_narrow_the_modes():
    with pytest.raises(ValueError, match="none, model"):
        offload_mode("meissonic", {"offload": "sequential"}, allowed=("none", "model"))


def test_old_yaml_keys_fail():
    with pytest.raises(ValueError, match="sequential_cpu_offload"):
        offload_mode("flux1-dev", {"sequential_cpu_offload": False})


def test_old_env_vars_fail(monkeypatch):
    monkeypatch.setenv("FLUX2_KLEIN_4B_SEQUENTIAL_OFFLOAD", "false")
    with pytest.raises(ValueError, match="FLUX2_KLEIN_4B_OFFLOAD"):
        offload_mode("flux2-klein-4b", {"offload": "none"})
