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
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import _PHASE_DECODE, SamplingParams, build_engine
from tilerl.spec import LADDER_WIDTHS, load_draft

DEPTHS = (1, 2, 3, 4)


def set_depth_in_place(e, head, depth: int) -> None:
    """Move a live engine to `depth`, and FAIL if the move did not take.

    The sweep's whole output is a difference between depths, so a depth knob that
    silently does nothing does not produce a wrong number -- it produces four
    copies of one config and a draft cost near zero, which reads as "drafting is
    free" rather than as a broken script. That is what happened here: this line
    used to be `e._spec_depth = d`, live at engine.py:893 until 7069a1f moved the
    chain loop onto the head, after which nothing read the attribute.

    Both the head and the engine hold the width, and the engine's copy is what a
    tick keys on (engine.py:860 graph_keys, :440 the KV reserve), so both move.
    """
    head.set_depth(depth)
    e._width = head.width
    assert head.width == depth + 1, f"head kept width {head.width} for depth {depth}"
    assert e._width == depth + 1, f"engine kept width {e._width} for depth {depth}"


def measure(e, ctx: int, tokens: int, vocab: int) -> tuple[float, float, float]:
    """(ms per decode tick, tokens per forward, mean chain width) over DECODE ticks.

    The prompt is drawn from the whole vocabulary, not `range(10, 10+ctx)`. That
    old prompt makes tok/forward a function of ctx -- it reads a different slice of
    the vocab at every length -- and at ctx=1024 it inflated acceptance from 2.03
    to 2.86, which is 1.409x and straddles the 2.776 break-even at W=4.
    errors/2026-09-03-the-context-sweep-changed-the-prompt.md

    Only the SECOND return value is affected. ms/tick is a cost, measured at a
    fixed rung, and does not depend on which tokens are in the prompt -- so the
    draft-share subtraction this script exists for stands either way.

    The THIRD is what tells you the subtraction is legal at all. `verify_lens`
    trims the chain per tick from the draft's confidences, so a configured depth
    is an upper bound on the width a tick actually verifies -- and this script's
    premise is that depths 2 and 3 land on the SAME rung. Measured, not assumed.
    """
    rid = e.submit(torch.randint(0, vocab, (ctx,),
                                 generator=torch.Generator().manual_seed(1000)).tolist(),
                   SamplingParams(temperature=0.0, max_new_tokens=tokens, seed=0))
    req = None
    while req is None or req.phase != _PHASE_DECODE:
        e.step()
        req = next((r for r in e._running if r.req_id == rid), None)
        if req is None:
            raise SystemExit(f"ctx={ctx}: finished during prefill")
    torch.cuda.synchronize()
    s0, t0 = e.stats(), time.perf_counter()
    out, per_rung = None, defaultdict(list)
    while out is None:
        b0, k0 = e.stats(), time.perf_counter()
        e.step()
        b1 = e.stats()
        # Time and price each tick by ITS OWN rung, not by the rung of the mean.
        # A mean chain of 2.56 is 72% rung-2 ticks and 28% rung-4; depth 3's 2.88
        # is 56/44. So 0.16 of all ticks change rung between the two depths, and
        # rung 2 -> 4 is 10.54 ms -- up to 1.69 of the 5.01 ms "one draft forward"
        # is that mix moving, not a draft. Rounding the mean cannot see it.
        # No per-tick sync: that would serialize what the tick overlaps and change
        # the cost being measured. The deltas are therefore submission-to-
        # submission, which is what a serving loop sees, and the window total
        # below is still the synced number.
        if b1["decode_forwards"] > b0["decode_forwards"]:
            w = 1 + b1["spec_drafted"] - b0["spec_drafted"]
            per_rung[next(r for r in LADDER_WIDTHS if r >= w)].append(
                (time.perf_counter() - k0) * 1000)
        out = e.poll().get(rid)
    torch.cuda.synchronize()
    wall, s1 = (time.perf_counter() - t0) * 1000, e.stats()
    n = s1["tokens_generated"] - s0["tokens_generated"]
    fwd = s1["decode_forwards"] - s0["decode_forwards"]
    if s1["mixed_forwards"] - s0["mixed_forwards"]:
        raise SystemExit(f"ctx={ctx}: mixed tick inside the window")
    width = 1 + (s1["spec_drafted"] - s0["spec_drafted"]) / max(fwd, 1)
    return wall / max(fwd, 1), n / max(fwd, 1), width, dict(per_rung)


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

    print(f"# ctx={args.ctx}. Configured width is depth+1; rungs are {LADDER_WIDTHS}.")
    print(f"# {'depth':>5} {'W':>3} {'chain':>6} {'ms/tick':>8} "
          f"{'tok/fwd':>8} {'tok/s':>7}  per-rung: rNxCOUNT:MEAN_MS")
    rows = {}
    # ONE engine, depth varied in place. A fresh engine per depth OOMs: the KV
    # pool and captured graphs outlive shutdown() (which only joins the daemon
    # thread), and each build re-quantizes the draft into new tensors. The graph
    # is captured per (batch, chain width), so each depth is warmed before it is
    # timed and the capture stays outside the window.
    e = build_engine(cfg, model, be, num_blocks=1024, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft, spec_depth=max(DEPTHS))
    for d in DEPTHS:
        set_depth_in_place(e, draft, d)
        measure(e, args.ctx, args.tokens, cfg.vocab_size)  # warm: JIT + graph capture
        ms, tpf, chain, per_rung = measure(e, args.ctx, args.tokens, cfg.vocab_size)
        rows[d] = (ms, tpf, chain, per_rung)
        mix = " ".join(f"r{r}x{len(v)}:{sum(v)/len(v):.1f}"
                       for r, v in sorted(per_rung.items()))
        print(f"{d:>5} {1 + d:>3} {chain:>6.2f} {ms:>8.2f} {tpf:>8.2f} "
              f"{1000 * tpf / ms:>7.1f}  {mix}")
    e.shutdown()

    # One draft forward, priced WITHIN one rung. Depths 2 and 3 both configure
    # width 3/4 -> rung 4, but verify_lens trims per tick, so each depth actually
    # runs a MIX: at ctx=1024 depth 2 was 72% rung-2 ticks and depth 3 was 56%.
    # Subtracting the two depths' MEAN ticks therefore moves 16% of all ticks
    # across a 10.54 ms rung step and charges it to the draft -- up to 1.69 of a
    # 5.01 ms number. So compare rung-4 ticks at depth 3 against rung-4 ticks at
    # depth 2: same verify shape on both sides, and the only difference left is
    # the one extra draft forward. Same failure as
    # wins/2026-09-03-long-context-decode-is-all-tick-cost.md, one level finer:
    # there the rung moved between contexts, here between depths.
    RUNG = 4
    have = {d: rows[d][3].get(RUNG, []) for d in (2, 3)}
    if min(len(v) for v in have.values()) < 5:
        raise SystemExit(
            f"too few rung-{RUNG} ticks to price a draft forward: depth 2 has "
            f"{len(have[2])}, depth 3 has {len(have[3])} -- verify_lens trimmed "
            "almost everything, so raise --tokens or lower --ctx")
    t2, t3 = (sum(v) / len(v) for v in (have[2], have[3]))
    draft = t3 - t2
    verify = t2 - 2 * draft
    print(f"\nrung-{RUNG} ticks only ({len(have[2])} at depth 2, {len(have[3])} at "
          f"depth 3): {t3:.2f} - {t2:.2f} = {draft:.2f} ms per draft forward")
    print(f"  a rung-{RUNG} depth-3 tick = {verify:.2f} verify + 3 x {draft:.2f} draft")
    print(f"  drafting is {100 * 3 * draft / t3:.0f}% of it")
    # The mean-tick subtraction the previous version reported, kept as the
    # contamination measurement rather than deleted: the gap between the two IS
    # the rung mix, and printing both is what makes that visible.
    ms2, ms3, tpf3 = rows[2][0], rows[3][0], rows[3][1]
    print(f"  (mean-tick subtraction: {ms3:.2f} - {ms2:.2f} = {ms3 - ms2:.2f} ms, "
          f"{ms3 - ms2 - draft:+.2f} of which is the rung mix moving)")
    # Verify cost per rung, measured directly now that ticks are bucketed: each
    # rung's mean tick minus its own draft forwards. These must form a staircase,
    # and it is an independent check on the one draft number above.
    print("  verify by rung: " + ", ".join(
        f"r{r}@d{d}: {sum(v)/len(v) - d * draft:.2f} ({len(v)})"
        for d in sorted(rows) for r, v in sorted(rows[d][3].items()) if len(v) >= 3))

    # A block-parallel head emits the whole block in ONE forward, so it removes
    # (D-1) draft forwards and changes nothing else. Priced on rung-4 ticks, the
    # same population the draft number came from.
    ideal = t3 - 2 * draft
    ceiling = t3 / ideal
    print("\nCEILING for a block-parallel draft head (1 forward instead of 3):")
    print(f"  {t3:.2f} -> {ideal:.2f} ms/tick, {ceiling:.3f}x at the same tok/forward")
    print("Upper bound: it assumes a parallel head drafts as well as the")
    print("autoregressive one, and a parallel position cannot see what was")
    print("sampled before it, so every point of accuracy lost cuts tok/fwd.")
    # The verdict is a RATIO and the prompt cancels out of it. Quoting a
    # break-even in tok/forward invites comparing it against an acceptance
    # measured on a different prompt, which is how a 1.111x arm got recorded as
    # 0.675x: break-even scales WITH tok/forward (= tpf x ideal/t3), so only
    # `yield / tok_fwd x ceiling` is prompt-independent.
    print(f"  a parallel head wins iff it keeps > {100 / ceiling:.1f}% of the "
          f"autoregressive head's tok/forward, whatever that is on your prompt")
    print(f"  (at the {tpf3:.2f} measured here that is {tpf3 / ceiling:.2f} tok/fwd)")


if __name__ == "__main__":
    main()
