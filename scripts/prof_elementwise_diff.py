"""Locate the torch elementwise launches by DIFFERENCING configurations.

One depth-3 tick spends 9.1 ms across ~1314 torch elementwise/index kernels —
17% of GPU-busy, none of it TileLang work. `with_stack=True` returns no Python
frames on this build (torch 2.5.1+cu121), so attribution has to come from
differences between runs that isolate one part of the model each.

Three counts, each a steady tick:

  dense          trunk only, one row              -> per-forward baseline
  spec depth 1   + one draft forward              -> one draft forward's share
  spec depth 3   + three                          -> linear in depth?

spec(d1) - dense isolates one draft forward including its bookkeeping; spec(d3) -
spec(d1) over 2 checks that the per-forward cost is constant, which is the test
that the difference means what it claims. What is left in `dense` is the trunk's
own, and that is the number that says whether the fix belongs in the draft path or
in model.py/backend.py.

Counts, not wall time: a launch count is exact and survives the profiler's
distortion (it reads 121.5 ms/tick where the tick is 66.46), where a wall-clock
ratio does not. GPU us per kernel is a sum of durations and also survives.
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

TORCH_MARKS = ("elementwise_kernel", "index_elementwise", "unrolled_elementwise",
               "vectorized_elementwise", "reduce_kernel", "CatArrayBatched")


def count(e, reps: int = 2) -> dict:
    """Launch counts and GPU us for one steady tick, split torch vs tilelang."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):  # settle
        e.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            e.step()
        torch.cuda.synchronize()
    out: dict = defaultdict(lambda: [0, 0.0])
    for ev in prof.events():
        if ev.device_type.name != "CUDA":
            continue
        us = ev.time_range.elapsed_us() / reps
        torchy = any(m in ev.name for m in TORCH_MARKS)
        out["torch" if torchy else "tilelang"][0] += 1 / reps
        out["torch" if torchy else "tilelang"][1] += us
        out[ev.name[:44]][0] += 1 / reps
        out[ev.name[:44]][1] += us
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True, source=args.source)

    cases = [("dense", 1, False), ("spec d1", 1, True), ("spec d3", 3, True)]
    got = {}
    draft = None
    for label, depth, wd in cases:
        # load_draft AFTER the dense case: build_engine does
        # `model.params = backend.materialize(model.params)` (engine.py:1139), which
        # rebinds the dict. A draft loaded before that keeps its own params over the
        # pre-materialize trunk, and the first lookup that misses raises
        # `KeyError: 'fc'` — which is what this script did on its first run.
        if wd and draft is None:
            draft = load_draft(model, args.draft)
        # ONE engine per case, and it is never shut down: shutdown() only joins a
        # daemon thread we never started, so a fresh engine per case OOMs the card
        # (errors/2026-09-02-synchronize-inside-a-captured-graph.md). Instead the
        # pools are the same size every time and the old engine is dropped before
        # the next is built, with an empty_cache between.
        e = build_engine(cfg, model, backend, num_blocks=512, num_slots=4, max_batch=4,
                         max_total_tokens=8192,
                         draft=draft if wd else None, spec_depth=depth)
        rid = e.submit(list(range(10, 40)),
                       SamplingParams(temperature=0.0, max_new_tokens=64, seed=0))
        while not any(r.req_id == rid and r.phase == 2 for r in e._running):
            e.step()
        got[label] = count(e)
        del e
        torch.cuda.empty_cache()

    print(f"\n{'case':<10} {'torch n':>9} {'torch ms':>9} {'tl n':>7} {'tl ms':>8}")
    for label, _, _ in cases:
        g = got[label]
        print(f"{label:<10} {g['torch'][0]:>9.0f} {g['torch'][1] / 1e3:>9.2f} "
              f"{g['tilelang'][0]:>7.0f} {g['tilelang'][1] / 1e3:>8.2f}")

    d, s1, s3 = (got[k]["torch"] for k in ("dense", "spec d1", "spec d3"))
    print(f"\none draft forward adds  {s1[0] - d[0]:>6.0f} torch launches, "
          f"{(s1[1] - d[1]) / 1e3:>5.2f} ms")
    print(f"each further forward    {(s3[0] - s1[0]) / 2:>6.0f} torch launches, "
          f"{(s3[1] - s1[1]) / 2e3:>5.2f} ms")
    print(f"dense trunk alone       {d[0]:>6.0f} torch launches, {d[1] / 1e3:>5.2f} ms")
    print("\nIf the trunk dominates, the fix is in model.py/backend.py, not the draft.")

    print(f"\n{'top torch kernels (spec d3)':<46} {'n':>7} {'ms':>8}")
    for name, (c, us) in sorted(got["spec d3"].items(), key=lambda kv: -kv[1][1]):
        if name in ("torch", "tilelang") or not any(m in name for m in TORCH_MARKS):
            continue
        print(f"{name:<46} {c:>7.0f} {us / 1e3:>8.3f}")


if __name__ == "__main__":
    main()
