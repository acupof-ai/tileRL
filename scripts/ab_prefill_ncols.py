"""In-process ncols A/B on the real 27B prefill path, plus a first-token parity check.

Two separate processes are not an A/B: the cross-process run of bench_prefill.py
reported ncols=1 at 2.34 ms/token @4096 against a recorded 8.92 baseline for the
same code — 3.8x FASTER than the kernel it is supposed to be. Whatever that was,
it was not the flag, so this builds ONE model and ONE engine and flips
backend._NCOLS between measurements. Same weights, same clock, same page cache,
one variable.

Both variants are compiled and cached (the kernel cache keys on nc), and each ctx
is measured nc=2, nc=1, nc=2 so a monotone drift shows up as the two nc=2 readings
disagreeing rather than as a fake gain.

TIMING ONLY. The token ids this prints are NOT a correctness check: the prompts are
synthetic (`range(base, base+ctx)`), and greedy over their logits returned id 0 in
every arm at every context -- including the run whose kernels were deliberately
wrong. Real text parity lives in scripts/parity_ncols.py; run it too.

  scripts/v100.sh run pfab2 'CKPT=...; /usr/bin/python3 -u scripts/ab_prefill_ncols.py \
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
from tilerl_kernels import backend as bk_mod
from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import _PHASE_DECODE, SamplingParams, build_engine

CTXS = (512, 2048, 4096)
#: nc=2 twice, so drift is visible as disagreement between them, not as a gain.
ARMS = (2, 1, 2)


def one(e, ctx: int, seq: int) -> tuple[float, int]:
    """Time submit-to-first-token and return (ms, prefill ticks)."""
    base = 10 + seq * 100000 + ctx  # distinct tokens: a repeat would hit the prefix cache
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
    out = None
    while out is None:  # drain so the slot frees
        e.step()
        out = e.poll().get(rid)
    return ms, ticks


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

    seq = 0
    for ctx in CTXS:  # warm: JIT + graph capture for BOTH variants, untimed
        for nc in (2, 1):
            bk_mod._NCOLS = nc
            one(e, ctx, seq)
            seq += 1

    print(f"\n# in-process ncols A/B, {'ctx':>6} " +
          " ".join(f"{'nc' + str(nc) + ' ms/tok':>13}" for nc in ARMS))
    for ctx in CTXS:
        best = []
        for nc in ARMS:
            bk_mod._NCOLS = nc
            b = None
            for _ in range(args.reps):
                ms, ticks = one(e, ctx, seq)
                seq += 1
                b = ms if b is None or ms < b else b
            best.append(b / ctx)
        print(f"{'':>19} {ctx:>6} " + " ".join(f"{v:>13.2f}" for v in best))
        print(f"{'':>26} ticks {ticks}, nc2 self-consistency "
              f"{best[0] / best[2]:.3f}x, nc1/nc2 {best[1] / best[0]:.3f}x")
    print("\nTiming only -- correctness is scripts/parity_ncols.py (real text, greedy).")


if __name__ == "__main__":
    main()
