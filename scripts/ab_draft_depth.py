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
from tilerl.kv_cache import BLOCK_TOKENS
from tilerl.spec import LADDER_WIDTHS, load_draft
from tilerl.tokenizer import get_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import wikitext_ids  # noqa: E402  (after the sys.path insert above)

DEPTHS = (1, 2, 3, 4)


def _sync() -> None:
    """Bracket a timed window, on any target. Unguarded `torch.cuda.synchronize()`
    raises "Torch not compiled with CUDA enabled" on the CPU cell, so the batching
    logic here could only be exercised on the pod."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_state() -> str:
    """SM clock, throttle reasons and load, sampled per depth.

    Two runs of the identical command on the same commit agreed to 0.1% at depths
    1-2 and diverged 1.58x and 2.08x at depths 3-4, with the SAME tick counts --
    same work, twice the wall time, only later in the run. A tick's wall is ~35 ms
    longer than its GPU time (the draft runs outside the graph), so anything that
    slows the host or the clock lands directly in these numbers. Sampling it makes
    a contaminated run say so instead of publishing a depth curve.
    """
    import subprocess
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks_throttle_reasons.active,"
             "temperature.gpu,power.draw", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:  # a probe must never take the measurement down
        q = f"unavailable: {exc}"
    return f"{q} load={os.getloadavg()[0]:.2f}"


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


def measure(e, prompts: list[list[int]], tokens: int) -> tuple[float, float, float, dict]:
    """(ms per decode tick, tokens per forward, mean rows x width, per-rung times).

    All of `prompts` are submitted before the first step, so `len(prompts)` is the
    batch: the depth question is not scale-free on this arch. `_NCOLS_MIN_M = 32`
    switches the verify GEMV to ncols=2 at M = rows x width >= 32, and the ladder
    rounds M up, so B decides which kernel each depth runs on AND how much of the
    rung is padding. Enumerated (M, rung, kernel):

        B=1   d1 (2,2,nc1)   d2 (3,4,nc1)   d3 (4,4,nc1)    d4 (5,8,nc1)
        B=2   d1 (4,4,nc1)   d2 (6,8,nc1)   d3 (8,8,nc1)    d4 (10,32,nc2)
        B=4   d1 (8,8,nc1)   d2 (12,32,nc2) d3 (16,32,nc2)  d4 (20,32,nc2)

    So at B=4 depth 1 fills rung 8 exactly on the slow kernel while depth 3 half-
    fills rung 32 on the fast one -- two variables move at once, and neither is the
    one that decided B=1. Past M=32 the engine chunks at 32 rather than climbing
    (engine.py:842), so B=8 d4 is 40 rows = two launches, not a rung.

    `prompt` content decides tok/forward and nothing else. ms/tick is a cost measured
    at a fixed rung and does not depend on which tokens are in the prompt, so the
    draft-share subtraction this script exists for holds on any prompt -- but the
    ACCEPTANCE half only means something with its distribution named. Random
    vocabulary reads 2.99 at W=4 and wikitext 2.36
    (wins/2026-09-04-depth-default-is-wrong-on-text.md), so a default settled on
    synthetic ids is settled on an artifact.

    The third value tells you the subtraction is legal at all: `verify_lens` trims
    the chain per tick from the draft's confidences, so a configured depth is an
    upper bound on the width a tick verifies, and each depth runs a MIXTURE of
    rungs. The fourth buckets ticks by their own M so the comparison stays inside
    one.
    """
    rids = [e.submit(list(p), SamplingParams(temperature=0.0, max_new_tokens=tokens,
                                             seed=0)) for p in prompts]
    while True:
        e.step()
        live = [r for r in e._running if r.req_id in rids]
        if len(live) == len(rids) and all(r.phase == _PHASE_DECODE for r in live):
            break
        if not live and e.poll():
            raise SystemExit(f"B={len(prompts)}: finished during prefill")
    _sync()
    s0, t0 = e.stats(), time.perf_counter()
    done, per_rung = {}, defaultdict(list)
    while len(done) < len(rids):
        b0, k0 = e.stats(), time.perf_counter()
        e.step()
        b1 = e.stats()
        # Time and price each tick by ITS OWN M = rows x width, not by the M of the
        # mean. At B=1 a mean chain of 2.56 was 72% rung-2 ticks and 28% rung-4 while
        # depth 3's 2.88 was 56/44, so 0.16 of all ticks changed rung between the two
        # depths and the 16.73 ms rung step landed inside a 5 ms "draft forward".
        # No per-tick sync: that would serialize what the tick overlaps and change
        # the cost being measured. The deltas are therefore submission-to-
        # submission, which is what a serving loop sees, and the window total
        # below is still the synced number.
        nf = b1["decode_forwards"] - b0["decode_forwards"]
        if nf:
            rows = len([r for r in e._running if r.req_id in rids and r.req_id not in done])
            w = 1 + (b1["spec_drafted"] - b0["spec_drafted"]) / max(rows, 1)
            m = max(rows, 1) * w
            per_rung[next(r for r in LADDER_WIDTHS if r >= m)].append(
                (time.perf_counter() - k0) * 1000 / nf)
        done.update({k: v for k, v in e.poll().items() if k in rids})
    _sync()
    wall, s1 = (time.perf_counter() - t0) * 1000, e.stats()
    n = s1["tokens_generated"] - s0["tokens_generated"]
    fwd = s1["decode_forwards"] - s0["decode_forwards"]
    if s1["mixed_forwards"] - s0["mixed_forwards"]:
        raise SystemExit(f"B={len(prompts)}: mixed tick inside the window")
    # Rows retire at different times, so tokens/forward is the batch's aggregate and
    # `m` below is the mean M actually launched, not rows x configured width.
    width = 1 + (s1["spec_drafted"] - s0["spec_drafted"]) / max(fwd, 1)
    return wall / max(fwd, 1), n / max(fwd, 1), width, dict(per_rung)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompt", choices=("wikitext", "random"), default="wikitext",
                    help="wikitext-103 test text (default: the only arm whose "
                         "acceptance means anything) or uniform ids over the vocab")
    ap.add_argument("--prompts", type=int, default=3,
                    help="distinct passages, cut into disjoint groups of --batch. Must "
                         "be a multiple of every --batch. Acceptance varies 15.8% "
                         "between wikitext passages, so one group measures one sample")
    ap.add_argument("--batch", default="1",
                    help="comma-separated batch sizes. The depth answer is NOT "
                         "scale-free: the verify GEMV switches kernel at M=rows x "
                         "width >= 32 and the ladder rounds M up, so at B=4 depth 1 "
                         "fills rung 8 with ncols=1 while depth 3 half-fills rung 32 "
                         "with ncols=2")
    args = ap.parse_args()
    batches = [int(b) for b in args.batch.split(",")]
    for B in batches:
        if args.prompts % B:
            raise SystemExit(
                f"--prompts {args.prompts} is not a multiple of --batch {B}: the run "
                f"would silently drop {args.prompts % B} passage(s) from the last group "
                "and compare batch sizes over different text")
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    be = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft)
    if args.prompt == "wikitext":
        prompts = wikitext_ids(get_tokenizer(args.source), args.prompts, args.ctx)
    else:
        g = torch.Generator().manual_seed(1000)
        prompts = [torch.randint(0, cfg.vocab_size, (args.ctx,), generator=g).tolist()
                   for _ in range(args.prompts)]

    # The token cap is a first-class parameter of the result, not a runtime knob: a
    # cap below the natural completion length truncates every row and changes what
    # is being compared, which is how a GSM8K arm read 38.5% against a recorded 85%.
    print(f"# ctx={args.ctx}, prompt={args.prompt} x{args.prompts}, "
          f"max_new_tokens={args.tokens}, ncols gate M>=32. "
          f"Configured width is depth+1; rungs are {LADDER_WIDTHS}.")
    rows = {}
    # ONE engine, depth varied in place. A fresh engine per depth OOMs: the KV
    # pool and captured graphs outlive shutdown() (which only joins the daemon
    # thread), and each build re-quantizes the draft into new tensors. The graph
    # is captured per (batch, chain width), so each depth is warmed before it is
    # timed and the capture stays outside the window.
    # Blocks sized to what the run needs, not a round number. The trunk pool AND the
    # draft's mirror of it are both num_blocks (spec.py:223), and at B=4 the flat 2048
    # spent 3.0 GiB of the two on 288 blocks of live context -- which is what the draft's
    # f32 prefill readout then could not find 1.88 GiB for.
    need = -(-(args.ctx + args.tokens) // BLOCK_TOKENS) * max(batches) + 8
    e = build_engine(cfg, model, be, num_blocks=need, num_slots=max(batches) + 1,
                     max_batch=max(batches), max_total_tokens=args.ctx + args.tokens + 64,
                     draft=draft, spec_depth=max(DEPTHS))
    for B in batches:
        print(f"\n# B={B}, M=rows x width -> {'/'.join(str(1 + d) for d in DEPTHS)} "
              f"x {B} = {'/'.join(str(B * (1 + d)) for d in DEPTHS)}, "
              f"{len(prompts) // B} disjoint passage group(s)"
              + ("  -- ONE group, so no between-passage spread is visible here"
                 if len(prompts) // B == 1 else ""))
        print(f"# {'depth':>5} {'W':>3} {'chain':>6} {'ms/tick':>8} "
              f"{'tok/fwd':>8} {'tok/s':>7}  per-M: rNxCOUNT:MEAN_MS")
        for d in DEPTHS:
            set_depth_in_place(e, draft, d)
            # DISJOINT groups of B, not `prompts[:B]` repeated: acceptance varies 15.8%
            # between wikitext passages (2.15 to 2.49 tok/fwd at W=4), so repeating one
            # passage prints four identical rows and reports run variance as if it were
            # corpus variance. Measured: ds10 read 2.49 four times where ds8, which
            # rotated, read 2.49/2.44/2.15 over the same three passages.
            groups = [prompts[i * B : (i + 1) * B] for i in range(len(prompts) // B)]
            measure(e, groups[0], args.tokens)  # warm: JIT + this (B, width) capture
            # Every depth sees the SAME groups in the same order, so a between-depth
            # difference cannot be a between-passage difference.
            got = [measure(e, g, args.tokens) for g in groups]
            # Per-group rows, not just their mean. Pooling hid a 1.58x drift: at
            # --prompts 3 on wikitext the pooled rung-4 mean read 97.5 ms against 61.8
            # for the first passage alone, which made verify come out NEGATIVE (-24 ms)
            # and the rate 24.3 tok/s against a known 45.9. A mean cannot show whether
            # the spread is between passages or along the run; these rows can.
            if len(got) > 1:
                for i, g in enumerate(got):
                    print(f"        p{i}: {g[0]:>8.2f} {g[1]:>8.2f} "
                          f"{1000 * g[1] / g[0]:>7.1f}")
            ms = sum(g[0] for g in got) / len(got)
            tpf = sum(g[1] for g in got) / len(got)
            chain = sum(g[2] for g in got) / len(got)
            per_rung = defaultdict(list)
            for g in got:
                for r, v in g[3].items():
                    per_rung[r].extend(v)
            rows[(B, d)] = (ms, tpf, chain, dict(per_rung))
            mix = " ".join(f"r{r}x{len(v)}:{sum(v)/len(v):.1f}"
                           for r, v in sorted(per_rung.items()))
            print(f"{d:>5} {1 + d:>3} {chain:>6.2f} {ms:>8.2f} {tpf:>8.2f} "
                  f"{1000 * tpf / ms:>7.1f}  {mix}")
            print(f"        gpu: {gpu_state()}")
        best = max(DEPTHS, key=lambda d: rows[(B, d)][1] / rows[(B, d)][0])
        r_best, r_ship = (1000 * rows[(B, d)][1] / rows[(B, d)][0] for d in (best, 3))
        print(f"  B={B}: best depth {best} at {r_best:.1f} tok/s, shipped 3 at "
              f"{r_ship:.1f} ({r_best / r_ship:.3f}x)"
              + ("  -- inside the 1.16% noise floor" if r_best / r_ship < 1.0116 else ""))
    e.shutdown()

    # One draft forward, priced WITHIN one rung, from the FIRST batch size only --
    # the draft-cost question is per-configuration and mixing batches would put two
    # kernels in one subtraction.
    B0 = batches[0]
    rows = {d: rows[(B0, d)] for d in DEPTHS}

    # Depths 2 and 3 configure width 3/4, so at B=1 both land on rung 4, but
    # verify_lens trims per tick and each depth actually runs a MIX -- measured at
    # ctx=1024 on random ids: depth 2 was 15 rung-2 ticks and 56 rung-4, depth 3 was
    # 14 and 55. Subtracting the two depths' MEAN ticks therefore drags part of that
    # mixture across the 16.73 ms rung step and charges it to the draft (0.61 of 4.54
    # ms there). So compare same-rung ticks on both sides: identical verify shape, and
    # the only difference left is the one extra draft forward. Same failure as
    # wins/2026-09-03-long-context-decode-is-all-tick-cost.md, one level finer: there
    # the rung moved between contexts, here between depths.
    #
    # The rung is derived from B, not hardcoded: at B=1 depths 2 and 3 share rung 4,
    # at B=4 they are M=12 -> rung 32 and M=16 -> rung 32 (shared), and at B=8 they
    # are M=24 and M=32 (also shared, but with different padding). Where they do NOT
    # share one, the subtraction is not a draft forward and the script says so.
    r2, r3 = (next(r for r in LADDER_WIDTHS if r >= B0 * (1 + d)) for d in (2, 3))
    if r2 != r3:
        raise SystemExit(
            f"at B={B0} depths 2 and 3 launch different rungs ({r2} vs {r3}), so "
            "their difference is a rung step, not one draft forward -- price the "
            "draft at a batch where they share a rung")
    RUNG = r3
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
