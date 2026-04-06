#!/usr/bin/env python3
"""Download model weights for InferGate.

Usage:
    python scripts/download_models.py --all
    python scripts/download_models.py --models flux1-schnell kokoro-82m
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_model_configs, load_server_config


def download_model(hub_id: str, model_dir: str, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    print(f"  Downloading {hub_id} -> {model_dir}")
    snapshot_download(
        repo_id=hub_id,
        cache_dir=model_dir,
        revision=revision or "main",
        token=token,
    )
    print(f"  Done: {hub_id}")


def main():
    parser = argparse.ArgumentParser(description="Download InferGate model weights")
    parser.add_argument("--all", action="store_true", help="Download all enabled models")
    parser.add_argument("--models", nargs="+", help="Specific model IDs to download")
    args = parser.parse_args()

    if not args.all and not args.models:
        parser.print_help()
        sys.exit(1)

    server_cfg = load_server_config()
    model_cfgs = load_model_configs()
    model_dir = server_cfg.models_dir

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    targets = model_cfgs if args.all else [c for c in model_cfgs if c.id in args.models]

    if not targets:
        print("No models to download.")
        sys.exit(0)

    print(f"Downloading {len(targets)} model(s) to {model_dir}/\n")
    for cfg in targets:
        if not cfg.enabled:
            print(f"[SKIP] {cfg.id} (disabled)")
            continue
        hub_id = cfg.model.get("hub_id")
        if not hub_id:
            print(f"[SKIP] {cfg.id} (no hub_id)")
            continue
        print(f"[{cfg.id}] {cfg.display_name}")
        try:
            download_model(hub_id, model_dir, cfg.model.get("revision"))
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
