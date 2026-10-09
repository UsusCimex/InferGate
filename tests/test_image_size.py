"""A request's size wins over the YAML width and height."""
from __future__ import annotations

from app.providers.image._size import apply_size


def test_size_overrides_yaml_dimensions():
    params = {"width": 1024, "height": 1024, "size": "512x768"}
    apply_size(params)
    assert params == {"width": 512, "height": 768}


def test_without_size_yaml_dimensions_stay():
    params = {"width": 1024, "height": 1024}
    apply_size(params)
    assert params == {"width": 1024, "height": 1024}
