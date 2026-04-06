#!/usr/bin/env python3
"""Simple benchmark for InferGate endpoints.

Usage:
    python scripts/benchmark.py --url http://localhost:8000 --endpoint chat --n 10
"""
from __future__ import annotations

import argparse
import statistics
import time

import httpx


def bench_chat(client: httpx.Client, n: int) -> list[float]:
    times = []
    for i in range(n):
        start = time.monotonic()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.5-9b",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 16,
            },
        )
        elapsed = time.monotonic() - start
        times.append(elapsed)
        status = resp.status_code
        print(f"  [{i+1}/{n}] {status} in {elapsed:.2f}s")
    return times


def bench_images(client: httpx.Client, n: int) -> list[float]:
    times = []
    for i in range(n):
        start = time.monotonic()
        resp = client.post(
            "/v1/images/generations",
            json={"prompt": "A red circle on white background", "size": "512x512"},
        )
        elapsed = time.monotonic() - start
        times.append(elapsed)
        print(f"  [{i+1}/{n}] {resp.status_code} in {elapsed:.2f}s")
    return times


def bench_tts(client: httpx.Client, n: int) -> list[float]:
    times = []
    for i in range(n):
        start = time.monotonic()
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "Hello, world!", "voice": "default"},
        )
        elapsed = time.monotonic() - start
        times.append(elapsed)
        print(f"  [{i+1}/{n}] {resp.status_code} in {elapsed:.2f}s")
    return times


def main():
    parser = argparse.ArgumentParser(description="InferGate benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--endpoint", choices=["chat", "images", "tts"], default="chat")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    client = httpx.Client(base_url=args.url, timeout=300)

    print(f"Benchmarking {args.endpoint} ({args.n} requests) at {args.url}\n")
    bench_fn = {"chat": bench_chat, "images": bench_images, "tts": bench_tts}[args.endpoint]
    times = bench_fn(client, args.n)

    if times:
        print(f"\nResults:")
        print(f"  Mean:   {statistics.mean(times):.3f}s")
        print(f"  Median: {statistics.median(times):.3f}s")
        print(f"  Min:    {min(times):.3f}s")
        print(f"  Max:    {max(times):.3f}s")
        if len(times) > 1:
            print(f"  Stdev:  {statistics.stdev(times):.3f}s")


if __name__ == "__main__":
    main()
