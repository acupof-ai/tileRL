"""Prefill wall time vs context — the path packed X above rung 8 actually touches.

Dense decode is M=1 and speculative verify is M=B*W, so neither reaches the
32-row rung; prefill does, on every layer of every chunk (M=512 at
max_num_batched_tokens=512). bench_ctx_decode.py deliberately excludes prefill
from its window, so it cannot see this.

Times the submit-to-first-token span with a cold prefix cache, which is prefill
plus one decode tick. Reports ms per prompt token so contexts are comparable.

  scripts/v100.sh run pp 'CKPT=...; /usr/bin/python3 -u scripts/bench_prefill.py \
      --source $CKPT'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import _PHASE_DECODE, SamplingParams, build_engine
from tilerl_kernels.backend import get_backend

CTXS = (512, 1024, 2048, 4096)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    be = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    e = build_engine(cfg, model, be, num_blocks=1024, num_slots=4, max_batch=4,
                     max_total_tokens=8192)

    print(f"# prefill to first token, {'ctx':>6} {'ms':>9} {'ms/tok':>8} {'ticks':>6}")
    for ctx in CTXS:
        best = None
        for rep in range(args.reps + 1):
            # Distinct tokens per rep: the prefix cache would serve a repeat from
            # HBM and time a lookup instead of a prefill.
            base = 10 + rep * 100000 + ctx
            rid = e.submit(list(range(base, base + ctx)),
                           SamplingParams(temperature=0.0, max_new_tokens=1, seed=0))
            torch.cuda.synchronize()
            s0, t0 = e.stats(), time.perf_counter()
            req = None
            while req is None or req.phase != _PHASE_DECODE:
                e.step()
                req = next((r for r in e._running if r.req_id == rid), None)
                if req is None:
                    break
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            ticks = e.stats()["prefill_forwards"] - s0["prefill_forwards"]
            while e.poll().get(rid) is None:  # drain so the slot frees
                e.step()
            if rep:  # rep 0 is JIT + graph capture
                best = (ms, ticks) if best is None or ms < best[0] else best
        ms, ticks = best
        print(f"{'':>24} {ctx:>6} {ms:>9.0f} {ms / ctx:>8.2f} {ticks:>6}")


if __name__ == "__main__":
    main()
