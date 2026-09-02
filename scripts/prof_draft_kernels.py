"""Per-kernel profile of ONE draft step, in the engine, not in isolation.

prof_spec_tick.py put 79% of a speculative tick in _draft_step at 371 ms/step
against 4.98 ms measured standalone. This says which kernels those 371 ms are.

  scripts/v100.sh run pd 'CKPT=...; /usr/bin/python3 -u scripts/prof_draft_kernels.py \
      --source $CKPT --draft $CKPT/model-00018-of-00018.safetensors --depth 3'
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.spec import load_draft
from tilerl_kernels.backend import get_backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft)
    e = build_engine(cfg, model, backend, num_blocks=512, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft, spec_depth=args.depth)
    rid = e.submit(list(range(10, 40)), SamplingParams(temperature=0.0, max_new_tokens=64, seed=0))
    for _ in range(4):  # past prefill, into steady speculative ticks
        e.step()

    import time

    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            e.step()
        torch.cuda.synchronize()
    reps = 3
    wall = (time.perf_counter() - t0) / reps * 1e3
    host: dict[str, list] = defaultdict(lambda: [0, 0.0])
    for ev in prof.events():
        if ev.device_type.name == "CPU":
            host[ev.name[:56]][0] += 1
            host[ev.name[:56]][1] += ev.self_cpu_time_total / reps
    by: dict[str, list] = defaultdict(lambda: [0, 0.0])
    tot = 0.0
    for ev in prof.events():
        if ev.device_type.name != "CUDA":
            continue
        us = ev.time_range.elapsed_us() / reps
        by[ev.name[:56]][0] += 1
        by[ev.name[:56]][1] += us
        tot += us
    n = sum(c for c, _ in by.values())
    print(f"\n=== one speculative tick (depth {args.depth}): GPU-busy {tot / 1e3:.1f} ms, "
          f"{n // reps} kernels ===")
    print(f"{'kernel':<56} {'n':>6} {'us ea':>8} {'ms':>8}")
    for name, (c, us) in sorted(by.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{name:<56} {c // reps:>6} {us / c * reps:>8.1f} {us / 1e3:>8.3f}")
    print(f"\nwall {wall:.1f} ms/tick, GPU-busy {tot / 1e3:.1f} ms -> "
          f"{100 * (1 - tot / 1e3 / wall):.0f}% host")
    print(f"\n{'host op (self CPU)':<56} {'n':>6} {'us ea':>8} {'ms':>8}")
    for name, (c, us) in sorted(host.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{name:<56} {c // reps:>6} {us / c * reps:>8.1f} {us / 1e3:>8.3f}")
    print(f"\n(rid={rid})")


if __name__ == "__main__":
    main()
