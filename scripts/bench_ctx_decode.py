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
from tilerl.cli import _build_model
from tilerl.engine import _PHASE_DECODE, SamplingParams, build_engine
from tilerl.spec import load_draft
from tilerl_kernels.backend import get_backend

CTXS = [32, 512, 1024, 2048, 4096]


def measure(e, ctx: int, tokens: int) -> tuple[float, float, int]:
    """tok/s and tok/forward over DECODE ticks only."""
    rid = e.submit(list(range(10, 10 + ctx)),
                   SamplingParams(temperature=0.0, max_new_tokens=tokens, seed=0))
    req = e._running[-1] if e._running else None
    while req is None or req.phase != _PHASE_DECODE:  # burn the prefill chunks
        e.step()
        req = next((r for r in e._running if r.req_id == rid), None)
        if req is None:
            raise SystemExit(f"ctx={ctx}: request finished during prefill")
    torch.cuda.synchronize()
    s0, t0 = e.stats(), time.perf_counter()
    out = None
    while out is None:
        e.step()
        out = e.poll().get(rid)
    torch.cuda.synchronize()
    wall, s1 = time.perf_counter() - t0, e.stats()
    n = s1["tokens_generated"] - s0["tokens_generated"]
    fwd = s1["decode_forwards"] - s0["decode_forwards"]
    mixed = s1["mixed_forwards"] - s0["mixed_forwards"]
    if mixed:  # a mixed tick never speculates; it would dilute tok/forward
        raise SystemExit(f"ctx={ctx}: {mixed} mixed ticks inside the window")
    return n / wall, n / max(fwd, 1), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft) if args.draft else None
    e = build_engine(cfg, model, backend, num_blocks=1024, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft,
                     spec_depth=args.depth if draft else 1)
    measure(e, 32, 32)  # warm every graph width before the first timed window

    label = f"spec d{args.depth}" if draft else "dense"
    print(f"\n{label}: {'ctx':>6} {'tok/s':>8} {'ms/tok':>8} {'tok/fwd':>8}")
    for ctx in CTXS:
        tps, per_fwd, _ = measure(e, ctx, args.tokens)
        print(f"{'':>{len(label) + 1}} {ctx:>6} {tps:>8.1f} {1000 / tps:>8.1f} {per_fwd:>8.2f}")


if __name__ == "__main__":
    main()
