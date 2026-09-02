"""How much of a speculative tick is the DRAFT, without instrumenting the tick?

The direct approach fails: prof_spec_tick.py wraps _run_forward with
cuda.synchronize(), which breaks CUDA-graph replay — it read 0.4 tok/s against a
real 48.4 and put 4972 ms in a 20 ms draft. Any probe inside a captured graph
measures the probe.

The indirect approach costs nothing and cannot lie. Depth D means D draft
forwards plus 1 verify per tick, so ms/tick is affine in D:

    ms_tick(D) = verify + D * draft

Run depth 1..4 at fixed context, regress, and the intercept is the verify and the
slope is one draft forward. tok/s falls out of ms/tick divided by the measured
tok/forward, which the engine already counts.

That number is the ceiling on block-parallel drafting (DFlash/DSpark): a head
emitting the whole block in ONE forward removes (D-1) * draft from every tick and
nothing else. If draft is a tenth of the tick, the idea is capped at ~10% no
matter how elegant.

  scripts/v100.sh run ds 'CKPT=...; /usr/bin/python3 -u scripts/ab_draft_depth.py \
      --source $CKPT --draft $CKPT/model-00018-of-00018.safetensors'
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

DEPTHS = (1, 2, 3, 4)


def measure(e, ctx: int, tokens: int) -> tuple[float, float]:
    """(ms per decode tick, tokens per forward) over DECODE ticks only."""
    rid = e.submit(list(range(10, 10 + ctx)),
                   SamplingParams(temperature=0.0, max_new_tokens=tokens, seed=0))
    req = None
    while req is None or req.phase != _PHASE_DECODE:
        e.step()
        req = next((r for r in e._running if r.req_id == rid), None)
        if req is None:
            raise SystemExit(f"ctx={ctx}: finished during prefill")
    torch.cuda.synchronize()
    s0, t0 = e.stats(), time.perf_counter()
    out = None
    while out is None:
        e.step()
        out = e.poll().get(rid)
    torch.cuda.synchronize()
    wall, s1 = (time.perf_counter() - t0) * 1000, e.stats()
    n = s1["tokens_generated"] - s0["tokens_generated"]
    fwd = s1["decode_forwards"] - s0["decode_forwards"]
    if s1["mixed_forwards"] - s0["mixed_forwards"]:
        raise SystemExit(f"ctx={ctx}: mixed tick inside the window")
    return wall / max(fwd, 1), n / max(fwd, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    be = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft)

    print(f"# ctx={args.ctx}, depth sweep. ms/tick is affine in depth:")
    print(f"# {'depth':>5} {'ms/tick':>8} {'tok/fwd':>8} {'tok/s':>7}")
    rows = []
    # ONE engine, depth varied in place. A fresh engine per depth OOMs: the KV
    # pool and captured graphs outlive shutdown() (which only joins the daemon
    # thread), and each build re-quantizes the draft into new tensors. The graph
    # is captured per (batch, chain width), so each depth is warmed before it is
    # timed and the capture stays outside the window.
    e = build_engine(cfg, model, be, num_blocks=1024, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft, spec_depth=max(DEPTHS))
    for d in DEPTHS:
        e._spec_depth = d
        measure(e, args.ctx, args.tokens)  # warm: JIT + this width's graph capture
        ms, tpf = measure(e, args.ctx, args.tokens)
        rows.append((d, ms, tpf))
        print(f"{d:>5} {ms:>8.2f} {tpf:>8.2f} {1000 * tpf / ms:>7.1f}")
    e.shutdown()

    # Least squares on ms = a + b*depth. b is one draft forward, a is everything
    # else in the tick (the verify trunk forward + sampling + bookkeeping).
    n = len(rows)
    sx = sum(d for d, _, _ in rows)
    sy = sum(ms for _, ms, _ in rows)
    sxx = sum(d * d for d, _, _ in rows)
    sxy = sum(d * ms for d, ms, _ in rows)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    print(f"\nms/tick = {a:.2f} + {b:.2f} * depth")
    print(f"  verify + overhead: {a:.2f} ms    one draft forward: {b:.2f} ms")

    d3 = next((r for r in rows if r[0] == 3), rows[-1])
    D, ms3, tpf3 = d3
    draft_share = 100 * b * D / ms3
    print(f"\nAt depth {D}: {b * D:.2f} of {ms3:.2f} ms is drafting = {draft_share:.0f}% of the tick.")
    # A block-parallel head emits the whole block in ONE forward, so it removes
    # (D-1) draft forwards and changes nothing else.
    ideal = ms3 - b * (D - 1)
    print(f"CEILING for a block-parallel draft head (one forward instead of {D}):")
    print(f"  {ms3:.2f} -> {ideal:.2f} ms/tick, {1000 * tpf3 / ms3:.1f} -> "
          f"{1000 * tpf3 / ideal:.1f} tok/s at the SAME {tpf3:.2f} tok/forward "
          f"({ms3 / ideal:.2f}x)")
    print("That is an upper bound: it assumes a parallel head drafts as well as")
    print("the autoregressive one. Every point of accuracy it loses cuts tok/fwd.")
    # Break-even: how much tok/forward a parallel head may lose before the
    # cheaper draft stops paying for itself.
    print(f"  break-even tok/forward: {tpf3 * ideal / ms3:.2f} "
          f"(below that, the current head wins)")


if __name__ == "__main__":
    main()
