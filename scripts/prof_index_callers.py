"""Which Python line issues the 252 index_elementwise launches in a spec tick?

Attribution of one depth-3 tick put them third-largest on the GPU at 3.97 ms
(7% of 58.5 ms GPU-busy) — larger than gdn_chunk_fused, and not model math. My
first guess (chain bookkeeping in _draft_step) accounts for ~10 of the 252, so it
was wrong by 25x. This finds the real call sites instead of guessing at them.

`with_stack=True` gives each kernel a Python traceback, so the launches can be
grouped by the source line that caused them. That is expensive and distorts the
wall clock badly — the wall is NOT a usable output here, only the grouping is.

  /usr/bin/python3 -u scripts/prof_index_callers.py --source $CKPT \
      --draft $CKPT/model-00018-of-00018.safetensors --depth 3
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl_kernels.backend import get_backend

from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.spec import load_draft

#: Kernels to attribute. Substring match on the profiler's name.
WANT = ("index_elementwise", "elementwise_kernel", "unrolled_elementwise", "vectorized_elementwise")

#: Frames in these files are plumbing, not the call site we want to blame.
SKIP = ("torch/", "site-packages/", "profiler", "prof_index_callers")


def blame(stack: list[str]) -> str:
    """The innermost frame that is our code, which is what we can change."""
    for fr in reversed(stack):
        if not any(s in fr for s in SKIP):
            return fr.split("/")[-1]
    return stack[-1].split("/")[-1] if stack else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--top", type=int, default=16)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True, source=args.source)
    draft = load_draft(model, args.draft) if args.draft else None
    e = build_engine(cfg, model, backend, num_blocks=512, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft,
                     spec_depth=args.depth if draft else 1)
    rid = e.submit(list(range(10, 40)), SamplingParams(temperature=0.0, max_new_tokens=64, seed=0))
    for _ in range(4):  # past prefill, into steady ticks
        e.step()

    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_stack=True) as prof:
        e.step()
        torch.cuda.synchronize()

    by: dict[str, list] = defaultdict(lambda: [0, 0.0])
    total = [0, 0.0]
    for ev in prof.events():
        if ev.device_type.name != "CUDA" or not any(w in ev.name for w in WANT):
            continue
        us = ev.time_range.elapsed_us()
        site = blame(list(getattr(ev, "stack", None) or []))
        by[site][0] += 1
        by[site][1] += us
        total[0] += 1
        total[1] += us

    print(f"\n=== elementwise/index launches in one tick: {total[0]}, "
          f"{total[1] / 1e3:.2f} ms ===")
    print(f"{'call site (innermost frame of ours)':<62} {'n':>5} {'ms':>8}")
    for site, (c, us) in sorted(by.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{site:<62} {c:>5} {us / 1e3:>8.3f}")
    if not by:
        print("(no stacks — this torch build may not carry them; fall back to "
              "counting call sites by hand)")
    print("\nWall clock under with_stack is meaningless; only the grouping is.")
    print(f"(rid={rid})")


if __name__ == "__main__":
    main()
