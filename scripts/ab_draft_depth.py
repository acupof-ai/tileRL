"""How much of a speculative tick is the DRAFT, without instrumenting the tick?

The direct approach fails: prof_spec_tick.py syncs around each wrapped method,
which breaks CUDA-graph replay — it read 0.4 tok/s against a real 48.4 and put
4972 ms in a 20 ms draft. Any probe inside a captured graph measures the probe.

The indirect approach has ONE trap, and it is the reason this file was rewritten:
depth changes TWO things at once. Depth D runs D draft forwards AND verifies a
chain of width D+1, and a verify row costs 10.7-14.1 ms. Regressing ms/tick on
depth across 1..4 therefore charges verify's growth to the draft — the same
"line through a staircase" error recorded in
errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md. A first version of
this script did exactly that (68%), and its supposed fix — restricting to depths
1 and 3 — compared rung 2 against rung 4 and was contaminated the same way (55%).

The clean comparison is depths 2 and 3: W=3 and W=4 BOTH dispatch rung 4
(spec.py:44 LADDER_WIDTHS = 1,2,4,8), so verify cost is identical and the only
difference is one draft forward. That single subtraction is the measurement.

    draft = ms_tick(3) - ms_tick(2)          # verify held constant at rung 4
    share = 2 * draft / ms_tick(2)           # depth 2 runs 2 draft forwards

Depths 1 and 4 are still run, as a cross-check the fit cannot fake: they must
come out ABOVE the rung-4 line, because W=2 drops a rung (cheaper verify) and
W=5 climbs to rung 8 (a cliff).

That number is the ceiling on block-parallel drafting (DFlash/DSpark): a head
emitting the whole block in ONE forward removes (D-1) draft forwards and nothing
else. If drafting is a tenth of the tick, the idea is capped at ~10%.

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
from tilerl.spec import LADDER_WIDTHS, load_draft
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

    print(f"# ctx={args.ctx}. Verify width is depth+1; rungs are {LADDER_WIDTHS}.")
    print(f"# {'depth':>5} {'W':>3} {'rung':>4} {'ms/tick':>8} {'tok/fwd':>8} {'tok/s':>7}")
    rows = {}
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
        rows[d] = (ms, tpf)
        rung = next(w for w in LADDER_WIDTHS if w >= 1 + d)
        print(f"{d:>5} {1 + d:>3} {rung:>4} {ms:>8.2f} {tpf:>8.2f} {1000 * tpf / ms:>7.1f}")
    e.shutdown()

    # Depths 2 and 3 (W=3, W=4) both dispatch rung 4, so verify is identical and
    # the difference is exactly one draft forward. No regression, no staircase.
    ms2, _ = rows[2]
    ms3, tpf3 = rows[3]
    draft = ms3 - ms2
    verify = ms2 - 2 * draft
    print(f"\nrung-4 pair: {ms3:.2f} - {ms2:.2f} = {draft:.2f} ms per draft forward")
    print(f"  depth 3 tick = {verify:.2f} verify + 3 x {draft:.2f} draft")
    print(f"  drafting is {100 * 3 * draft / ms3:.0f}% of a depth-3 tick")
    # Cross-check the pair against the depths it did NOT use, in the direction
    # each rung implies: depth 1 drops to rung 2 (a CHEAPER verify, so it must
    # land BELOW the rung-4 line) and depth 4 climbs to rung 8 (dearer, so
    # ABOVE). A depth that lands ON the line means the rungs are not what we
    # think — the point is that the two deviate in OPPOSITE directions, which a
    # single slope through all four cannot represent.
    for d, want in ((1, "below"), (4, "above")):
        if d in rows:
            pred = verify + d * draft
            got = "above" if rows[d][0] > pred else "below"
            print(f"  depth {d}: {rows[d][0]:.2f} vs {pred:.2f} rung-4 line — {got}"
                  f"{', expected' if got == want else ', SUSPECT (wanted ' + want + ')'}")
    # Verify cost per rung, derived from the ONE draft number: if 5.53 is right,
    # these must be a clean staircase, and that is an independent check on it.
    print(f"  verify by rung: " + ", ".join(
        f"w{1 + d}->{next(w for w in LADDER_WIDTHS if w >= 1 + d)}: {rows[d][0] - d * draft:.2f}"
        for d in sorted(rows)))

    # A block-parallel head emits the whole block in ONE forward, so it removes
    # (D-1) draft forwards and changes nothing else.
    ideal = ms3 - 2 * draft
    print(f"\nCEILING for a block-parallel draft head (1 forward instead of 3):")
    print(f"  {ms3:.2f} -> {ideal:.2f} ms/tick, {1000 * tpf3 / ms3:.1f} -> "
          f"{1000 * tpf3 / ideal:.1f} tok/s at the SAME {tpf3:.2f} tok/forward "
          f"({ms3 / ideal:.2f}x)")
    print("Upper bound: it assumes a parallel head drafts as well as the")
    print("autoregressive one, and a parallel position cannot see what was")
    print("sampled before it, so every point of accuracy lost cuts tok/fwd.")
    print(f"  break-even tok/forward: {tpf3 * ideal / ms3:.2f} "
          f"(below that, the current head wins)")


if __name__ == "__main__":
    main()
