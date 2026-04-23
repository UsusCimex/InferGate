"""Tests for ConfigWatcher — drive scan_once() directly to avoid sleep-loop races."""
from __future__ import annotations

import asyncio
import os

import pytest

from app.services.config_watcher import ConfigWatcher

_YAML_TEMPLATE = """\
id: {id}
display_name: {display_name}
category: image
provider_class: FakeImageProvider
enabled: true
model:
  hub_id: test/test
  vram_mb: 1000
"""


def _write_yaml(path, model_id="watch-test", display_name="first") -> None:
    path.write_text(_YAML_TEMPLATE.format(id=model_id, display_name=display_name))


@pytest.mark.asyncio
async def test_first_scan_is_baseline_only(tmp_path):
    """First scan snapshots mtimes without dispatching — otherwise every
    startup would re-reload every file."""
    yaml = tmp_path / "watch-test.yaml"
    _write_yaml(yaml)

    calls: list[str] = []

    async def cb(cfg):
        calls.append(cfg.id)

    watcher = ConfigWatcher(tmp_path, cb, debounce=0.0)
    changed = await watcher.scan_once()
    assert changed == []
    assert calls == []


@pytest.mark.asyncio
async def test_new_file_dispatches_reload(tmp_path):
    """A YAML appearing after the baseline scan fires the callback."""
    calls: list[str] = []

    async def cb(cfg):
        calls.append(cfg.display_name)

    watcher = ConfigWatcher(tmp_path, cb, debounce=0.0)
    await watcher.scan_once()  # baseline, empty dir

    _write_yaml(tmp_path / "watch-test.yaml", display_name="first")
    changed = await watcher.scan_once()
    assert len(changed) == 1
    assert calls == ["first"]


@pytest.mark.asyncio
async def test_modified_file_dispatches_reload(tmp_path):
    """mtime advancing past `debounce` triggers a re-dispatch."""
    yaml = tmp_path / "watch-test.yaml"
    _write_yaml(yaml, display_name="first")

    calls: list[str] = []

    async def cb(cfg):
        calls.append(cfg.display_name)

    watcher = ConfigWatcher(tmp_path, cb, debounce=0.0)
    await watcher.scan_once()  # baseline

    _write_yaml(yaml, display_name="second")
    # Bump mtime explicitly to avoid sub-second timestamp collisions on
    # filesystems with low mtime precision (HFS+, FAT32).
    old = os.stat(yaml).st_mtime
    os.utime(yaml, (old + 2, old + 2))

    changed = await watcher.scan_once()
    assert len(changed) == 1
    assert calls == ["second"]


@pytest.mark.asyncio
async def test_unchanged_file_is_noop(tmp_path):
    """Scanning twice with no edits must not fire the callback."""
    yaml = tmp_path / "watch-test.yaml"
    _write_yaml(yaml)

    calls: list[str] = []

    async def cb(cfg):
        calls.append(cfg.id)

    watcher = ConfigWatcher(tmp_path, cb, debounce=0.0)
    await watcher.scan_once()  # baseline
    changed = await watcher.scan_once()
    assert changed == []
    assert calls == []


@pytest.mark.asyncio
async def test_bad_yaml_does_not_crash_watcher(tmp_path, caplog):
    """Malformed YAML logs an error but leaves the watcher alive —
    must also NOT fire the callback with a half-parsed config."""
    yaml = tmp_path / "watch-test.yaml"
    _write_yaml(yaml)

    calls: list[str] = []

    async def cb(cfg):
        calls.append(cfg.id)

    watcher = ConfigWatcher(tmp_path, cb, debounce=0.0)
    await watcher.scan_once()  # baseline

    yaml.write_text("this: is: not: valid: yaml\n")
    os.utime(yaml, None)  # bump mtime to "now", past the baseline
    old = os.stat(yaml).st_mtime
    os.utime(yaml, (old + 2, old + 2))

    changed = await watcher.scan_once()
    # Path is reported as changed but parse failed → no callback
    assert len(changed) == 1
    assert calls == []


@pytest.mark.asyncio
async def test_callback_exception_does_not_block_other_files(tmp_path):
    """One file's failed reload must not starve subsequent files."""
    _write_yaml(tmp_path / "alpha.yaml", model_id="alpha")
    _write_yaml(tmp_path / "beta.yaml", model_id="beta")

    calls: list[str] = []

    async def cb(cfg):
        calls.append(cfg.id)
        if cfg.id == "alpha":
            raise RuntimeError("simulated callback failure")

    watcher = ConfigWatcher(tmp_path, cb, debounce=0.0)
    await watcher.scan_once()  # baseline

    # Bump both files' mtimes so both are "changed"
    for f in tmp_path.glob("*.yaml"):
        old = os.stat(f).st_mtime
        os.utime(f, (old + 2, old + 2))

    await watcher.scan_once()
    # Both callbacks ran — alpha raised but beta was still dispatched
    assert set(calls) == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_removed_file_logs_warning(tmp_path, caplog):
    """Deleted YAML logs a warning; callback not fired; watcher survives."""
    yaml = tmp_path / "watch-test.yaml"
    _write_yaml(yaml)

    calls: list[str] = []

    async def cb(cfg):
        calls.append(cfg.id)

    watcher = ConfigWatcher(tmp_path, cb, debounce=0.0)
    await watcher.scan_once()  # baseline snapshots the file

    yaml.unlink()
    with caplog.at_level("WARNING"):
        await watcher.scan_once()

    assert calls == []
    assert any("removed" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_start_stop_lifecycle(tmp_path):
    """start() spawns exactly one task; stop() cancels and awaits it."""
    async def cb(cfg):
        pass

    watcher = ConfigWatcher(tmp_path, cb, interval=60.0)
    assert watcher._task is None
    watcher.start()
    assert watcher._task is not None
    task = watcher._task
    # Double-start should be idempotent — no second task spawned.
    watcher.start()
    assert watcher._task is task
    await asyncio.sleep(0)  # let the task actually schedule
    await watcher.stop()
    assert watcher._task is None
    assert task.cancelled() or task.done()
