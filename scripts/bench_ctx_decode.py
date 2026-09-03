"""Steady-state decode rate vs context length — no prefill in the window.

The two-point slope in bench_workloads.py cannot measure long context: a 4K
prompt takes 8 chunked-prefill ticks at max_num_batched_tokens=512, speculation
is off on every mixed tick (engine.py:790), and at lo=32 those ticks dominate
the lo point instead of cancelling. That is why 4K read as 14.8 tok/s spec and
18.0 dense — both prefill, neither decode.

This drives the engine directly and times only ticks where the request is in
DECODE phase, so the window contains no prefill at all. Reports tok/s and
tokens per trunk forward against context length.

  scripts/v100.sh run lc 'CKPT=...; /usr/bin/python3 -u scripts/bench_ctx_decode.py \
      --source $CKPT [--draft $CKPT/model-00018-of-00018.safetensors]'
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
from tilerl.spec import load_draft
from tilerl_kernels.backend import get_backend

CTXS = [32, 512, 1024, 2048, 4096]


def measure(e, ctx: int, tokens: int, batch: int = 1) -> tuple[float, float, int]:
    """tok/s and tok/forward over DECODE ticks only.

    ``batch`` submits that many concurrent requests, which is the only way a tick
    reaches B*W rows: the engine's max_batch is an upper bound, so one request
    speculates at W rows and never touches the rungs serving actually compiles
    (B=4 W=4 is 16 rows -> rung 32, where ncols=2 turns on; B=1 W=4 is 4 rows and
    does not). A B=1 run silently measured the same kernel in both arms of the
    spec-ncols A/B -- errors/2026-09-03-the-spec-ncols-ab-ran-at-b1.md.
    """
    rids = [e.submit(list(range(10 + i * ctx, 10 + (i + 1) * ctx)),
                     SamplingParams(temperature=0.0, max_new_tokens=tokens, seed=0))
            for i in range(batch)]
    while True:  # burn the prefill chunks for every request
        reqs = [next((r for r in e._running if r.req_id == rid), None) for rid in rids]
        if all(r is not None and r.phase == _PHASE_DECODE for r in reqs):
            break
        if any(r is None for r in reqs) and e.poll():
            raise SystemExit(f"ctx={ctx}: a request finished during prefill")
        e.step()
    torch.cuda.synchronize()
    s0, t0 = e.stats(), time.perf_counter()
    done: dict = {}
    while len(done) < batch:
        e.step()
        done.update({k: v for k, v in e.poll().items() if k in rids})
    torch.cuda.synchronize()
    wall, s1 = time.perf_counter() - t0, e.stats()
    n = s1["tokens_generated"] - s0["tokens_generated"]
    fwd = s1["decode_forwards"] - s0["decode_forwards"]
    mixed = s1["mixed_forwards"] - s0["mixed_forwards"]
    if mixed:  # a mixed tick never speculates; it would dilute tok/forward
        raise SystemExit(f"ctx={ctx}: {mixed} mixed ticks inside the window")
    return n / wall, n / max(fwd, 1), n


def timed(e, ctx: int, tokens: int, batch: int = 1) -> tuple[float, float, str]:
    """Warm this context, then measure it, and flag an unwarmed reading.

    A speculative run captures a CUDA graph per (batch, chain width), and a
    width first seen inside a timed window puts its multi-second compile in the
    measurement — the third time that has silently ruined a number here. Two
    warmups, not one: the first also absorbs the kernel JIT that fires on the
    very first call at a new context. The ratio check then catches any capture
    that still slipped through, since a capture is seconds against a tick of
    tens of ms and a clean pair agrees closely.

    Flags rather than raises: a SystemExit here leaves the engine holding the
    whole card, and the orphan is invisible until the next run OOMs.
    """
    measure(e, ctx, tokens, batch)
    warm, _, _ = measure(e, ctx, tokens, batch)
    tps, per_fwd, _ = measure(e, ctx, tokens, batch)
    return tps, per_fwd, " UNWARMED" if tps > 2 * warm else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1,
                    help="concurrent requests per tick; the rung a verify tick compiles "
                         "keys on B*W, so B=1 never reaches the 32 rung serving uses")
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    # cli binds _QWEN38_SOURCE from the env at import, which already happened.
    cli._QWEN38_SOURCE = args.source

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft) if args.draft else None
    e = build_engine(cfg, model, backend, num_blocks=1024, num_slots=max(4, args.batch),
                     max_batch=max(4, args.batch),
                     max_total_tokens=8192, draft=draft,
                     spec_depth=args.depth if draft else 1)
    label = f"spec d{args.depth}" if draft else "dense"
    rows = args.batch * (1 + args.depth if draft else 1)
    print(f"\n{label} B={args.batch} ({rows} rows/tick): "
          f"{'ctx':>6} {'tok/s':>8} {'ms/tok':>8} {'tok/fwd':>8}")
    for ctx in CTXS:
        tps, per_fwd, flag = timed(e, ctx, args.tokens, args.batch)
        print(f"{ctx:>6} {tps:>8.1f} {1000 / tps:>8.1f} {per_fwd:>8.2f}{flag}")


if __name__ == "__main__":
    main()
