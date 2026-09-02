"""Qwen3.8-27B NVFP4 serving baseline: decode B=1 ms/tick and prefill tok/s
at 512 / 2048 / 8192 tokens through the serving build (fused, decode graph on).
    TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python3 scripts/bench_qwen38_baseline.py /data00/Qwen3.8-27B-NVFP4
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import load_hf
from tilerl_kernels.backend import get_backend


def _drive(engine, wid, max_steps) -> None:
    for _ in range(max_steps):
        engine.step()
        if wid in engine.poll():
            return
    raise RuntimeError("request did not finish")


def _rand_prompt(vocab: int, n: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=gen).tolist()


def time_decode(engine, vocab: int, ticks: int) -> tuple[float, float]:
    wid = engine.submit(
        _rand_prompt(vocab, 512, seed=11),
        SamplingParams(temperature=0.0, max_new_tokens=ticks + 4, seed=0),
    )
    engine.step()  # untimed prefill tick
    walls = []
    for _ in range(ticks):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.step()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
    _drive(engine, wid, 64)  # drain the last tokens
    ms = sum(walls) / len(walls) * 1000.0
    return ms, 1000.0 / ms


def time_prefill(engine, vocab: int, length: int, decode_ms: float) -> tuple[float, float]:
    wid = engine.submit(
        _rand_prompt(vocab, length, seed=length),
        SamplingParams(temperature=0.0, max_new_tokens=1, seed=0),
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _drive(engine, wid, max(1024, length // 128))
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    prefill_ms = total_ms - decode_ms
    return prefill_ms, 1000.0 * length / prefill_ms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="HF checkpoint directory")
    p.add_argument("--decode-ticks", type=int, default=32)
    args = p.parse_args()

    backend = get_backend()
    if backend.device.type != "cuda":
        raise SystemExit("baseline needs the CUDA target (TILERL_TARGET=cuda)")
    cfg = qwen38_27b()

    t0 = time.perf_counter()
    model = load_hf(cfg, args.source, fuse_projections=True)
    load_s = time.perf_counter() - t0
    print(f"load: {load_s:.1f}s", flush=True)

    engine = build_engine(
        cfg,
        model,
        backend,
        num_blocks=1024,  # serving: 256 (4096 tokens); bumped for 8192 prompts
        num_slots=16,
        max_batch=8,
        max_total_tokens=16384,  # serving: 8192
    )
    free, total = torch.cuda.mem_get_info()
    print(
        f"gpu: {torch.cuda.get_device_name(0)} | "
        f"mem after engine: {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB",
        flush=True,
    )

    # Prefill chunks are always M=512, so one 512-token prompt + decode compiles every shape.
    for pass_no in (1, 2):
        t0 = time.perf_counter()
        wid = engine.submit(
            _rand_prompt(cfg.vocab_size, 512, seed=pass_no),
            SamplingParams(temperature=0.0, max_new_tokens=4, seed=0),
        )
        _drive(engine, wid, 1024)
        print(f"warmup pass {pass_no}: {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"decode_graph_on: {engine._decode_graph_on}", flush=True)

    dec_ms, dec_tps = time_decode(engine, cfg.vocab_size, args.decode_ticks)
    print(f"\nDECODE B=1: {dec_ms:.2f} ms/tick | {dec_tps:.1f} tok/s", flush=True)

    print("\nPREFILL (single request, chunked at 512 tokens/tick)")
    print(f"  {'len':>6} {'ms/tok':>10} {'tok/s':>10}")
    for length in (512, 2048, 8192):
        ms, tps = time_prefill(engine, cfg.vocab_size, length, dec_ms)
        print(f"  {length:>6} {ms / length:>10.4f} {tps:>10.1f}", flush=True)

    free, _ = torch.cuda.mem_get_info()
    print(f"\npeak mem: {(torch.cuda.max_memory_allocated()) / 2**30:.1f} GiB", flush=True)
    print(f"bench_commit: {os.environ.get('BENCH_COMMIT', 'unknown')}", flush=True)
    print(f"target: {backend.target}", flush=True)
    print(f"engine stats: {engine.stats()}", flush=True)


if __name__ == "__main__":
    main()
