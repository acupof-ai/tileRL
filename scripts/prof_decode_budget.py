"""Per-kernel budget of one decode token: where do the 33 ms go?

Attention is now ~2.6 ms of a 33 ms token at 4096 ctx (wins/2026-09-01-sm70-
attention-thread-redundancy.md), so 30 ms is elsewhere and unattributed. The
weight roofline says 17.8 ms of it is unavoidable — the 16.04 GB a dense token
streams (trunk + lm_head) at 900 GB/s. This attributes the rest to kernels by
name, which is the step that turns "30 ms somewhere" into a target.

torch.profiler rather than another hand-rolled timer: the decode path is CUDA
-graph captured, and a graph replay shows up as its constituent kernels here
while manual event timing around the replay only gives the total.

  scripts/v100.sh run bud '/usr/bin/python3 -u scripts/prof_decode_budget.py'
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
from torch.profiler import ProfilerActivity, profile

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import _PHASE_DECODE, SamplingParams, build_engine
from tilerl.spec import LADDER_WIDTHS, load_draft
from tilerl.tokenizer import get_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import wikitext_ids  # noqa: E402  (after the sys.path insert above)

#: Kernel-name substring -> the op class it belongs to. First match wins, so
#: order matters: the fp4 GEMV names contain "gemv", attention contains "attn".
CLASSES = [
    ("attention", ("paged_attention", "attn")),
    ("fp4 GEMV", ("linear_fp4", "gemv")),
    ("GDN", ("gdn",)),
    ("rmsnorm", ("rmsnorm", "norm")),
    ("rope", ("rope",)),
    ("kv write", ("write_tokens", "scatter", "copy_kv")),
    ("sampler", ("softmax", "topk", "sort", "multinomial", "argmax")),
    ("elementwise", ("silu", "mul", "add", "cast", "convert", "elementwise")),
]


def classify(name: str) -> str:
    low = name.lower()
    for cls, keys in CLASSES:
        if any(k in low for k in keys):
            return cls
    return "other"


def comparable(width_lo: float, width_hi: float) -> str:
    """Is a two-context delta a context cost, or partly a verify-RUNG change?

    The load-bearing check in this script. At B=1 the rung is
    `next(w >= chain width)` and `verify_lens` trims the chain per tick from the
    draft's confidences, which rise with context -- so a longer context can draft
    wider and land on a dearer rung. The GEMV launch COUNT is per-layer and
    identical either way (measured: 330.2 vs 330.0 calls/forward across a 2->4
    crossing), so only M moves and the per-kernel table looks like the same work
    got 24% slower. It did not: 7.09 of a 10.79 ms/forward step was the rung.
    errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md
    """
    lo, hi = (next(w for w in LADDER_WIDTHS if w >= round(x)) for x in (width_lo, width_hi))
    if lo != hi:
        return (f"!! mean chain {width_lo:.2f} -> {width_hi:.2f} crosses a rung "
                f"({lo} -> {hi}): the deltas mix rung cost with context cost")
    return (f"#  mean chain {width_lo:.2f} -> {width_hi:.2f}, both on rung {lo} — "
            "same GEMV shapes, so these deltas are context")


def _self_check() -> None:
    """`comparable` must flag a rung crossing and pass a same-rung pair.

    Runs on the GPU-less box, because getting this wrong is not a crash: it is a
    plausible per-kernel table that attributes a rung to the context.
    """
    assert comparable(2.28, 3.70).startswith("!!"), "the measured 2->4 crossing must flag"
    assert comparable(1.00, 1.00).startswith("#"), "the dense control must pass"
    assert comparable(3.70, 3.90).startswith("#"), "same rung, different width: comparable"
    # Rounding is part of the rung, not a detail: 2.4 verifies on rung 2 and 2.6 on 4.
    assert comparable(2.40, 2.60).startswith("!!"), "a crossing inside one integer must flag"
    print("prof_decode_budget: comparable OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--ctx", default="4096",
                    help="comma-separated contexts, profiled in ONE process so the "
                         "pool, the captured graphs and the allocator are identical")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--prompt", choices=("random", "wikitext"), default="random",
                    help="uniform ids over the vocab, or wikitext-103 test text")
    ap.add_argument("--depths", default="",
                    help="comma-separated spec depths, profiled in ONE process at the "
                         "FIRST --ctx. This is the rung-step instrument: a verify at "
                         "rung W streams the same weights as W=1, only X grows, so a "
                         "bandwidth-bound GEMV should be nearly flat in M. Measured "
                         "end-to-end it is not (5.68/8.06/8.71 ms per extra row across "
                         "rungs 1->2->4->8), and per-kernel is where that gets "
                         "attributed. Needs --draft")
    args = ap.parse_args()
    ctxs = [int(c) for c in args.ctx.split(",")]
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft) if args.draft else None
    e = build_engine(cfg, model, backend, num_blocks=1024, num_slots=4, max_batch=4,
                     max_total_tokens=8192, draft=draft, spec_depth=3 if draft else 1)

    def to_decode(ctx: int, tokens: int):
        """Submit and burn the prefill chunks; return the request id.

        The default prompt is drawn from the whole vocabulary, not
        `range(10, 10+ctx)`: that old prompt makes its own CONTENT a function of
        ctx, so tok/forward moves with length for a reason that is not length
        (errors/2026-09-03-the-context-sweep-changed-the-prompt.md). Here it would
        put a different acceptance in each row's per-token divisor.

        `--prompt wikitext` exists because the two prompts do not just accept
        differently, they cost differently: at ctx=1024 depth 3 reads 97.5 ms per
        rung-4 tick on wikitext against 63.9 on random ids, same rung and same
        binary. Same-rung ticks should cost the same regardless of content, so one
        of those is measuring something other than the tick.
        """
        if args.prompt == "wikitext":
            ids = wikitext_ids(get_tokenizer(args.source), 1, ctx)[0]
        else:
            ids = torch.randint(0, cfg.vocab_size, (ctx,),
                                generator=torch.Generator().manual_seed(1000)).tolist()
        rid = e.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=tokens, seed=0))
        req = None
        while req is None or req.phase != _PHASE_DECODE:
            e.step()
            req = next((r for r in e._running if r.req_id == rid), None)
            if req is None:
                raise SystemExit(f"ctx={ctx}: finished during prefill")
        return rid

    def drain(rid: int) -> tuple[int, int, float]:
        """(tokens, decode forwards, mean drafted width) -- forwards is the divisor.

        ms/token folds acceptance into a cost: at the same tick a row accepting
        2.10 tok/forward reads cheaper per token than one accepting 2.03. Tick
        cost is what varies with context here, so normalize by forwards.

        The third value is why a per-kernel table alone cannot answer this. At
        B=1 the verify rung is `next(w >= chain width)` and `verify_lens` trims
        the chain per tick from the draft's confidences, so a context that drafts
        wider runs more rung-4 ticks and fewer rung-2 ones. GEMV launch COUNT is
        per-layer and identical either way; only M moves. Comparing two contexts
        without this is the staircase error in
        errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md.
        """
        s0 = e.stats()
        out = None
        while out is None:
            e.step()
            out = e.poll().get(rid)
        s1 = e.stats()
        if s1["mixed_forwards"] - s0["mixed_forwards"]:
            raise SystemExit("a mixed tick landed inside the window")
        fwd = s1["decode_forwards"] - s0["decode_forwards"]
        return (s1["tokens_generated"] - s0["tokens_generated"], fwd,
                1 + (s1["spec_drafted"] - s0["spec_drafted"]) / max(fwd, 1))

    def profile_one(ctx: int):
        """One profiled decode window at `ctx`. Returns the row tuple."""
        rid = to_decode(ctx, args.tokens)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            n, fwd, width = drain(rid)
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000 / max(fwd, 1)

        by_cls: dict[str, float] = defaultdict(float)
        by_kernel: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
        for ev in prof.key_averages():
            us = getattr(ev, "self_device_time_total", 0) or 0
            if us <= 0:
                continue
            by_cls[classify(ev.key)] += us
            ms, cnt = by_kernel[ev.key]
            by_kernel[ev.key] = (ms + us, cnt + ev.count)
        return (by_cls, by_kernel, n, fwd, wall_ms, width)

    for ctx in ctxs:
        for _ in range(2):  # warm: JIT, this width's graph capture, allocator
            drain(to_decode(ctx, args.tokens))
    torch.cuda.synchronize()

    # Profile the DECODE ticks only. A 4096 prompt is 8 chunked-prefill ticks
    # and they dwarf a decode tick, so profiling across them reported 8217
    # ms/token and 2900 GEMV calls/token — the prefill's work over the decode's
    # token count.
    rows = {ctx: profile_one(ctx) for ctx in ctxs}

    label = "spec d3" if draft else "dense"
    for ctx, (by_cls, by_kernel, n, fwd, wall_ms, width) in rows.items():
        total = sum(by_cls.values())
        gpu_ms = total / 1000 / max(fwd, 1)
        rung = next(w for w in LADDER_WIDTHS if w >= round(width))
        print(f"\n# {label}, ctx={ctx}, {n} tokens over {fwd} forwards "
              f"({n / max(fwd, 1):.2f} tok/fwd, mean chain {width:.2f} -> rung ~{rung})")
        print(f"# {gpu_ms:.2f} ms/FORWARD GPU vs {wall_ms:.2f} ms/forward wall")
        print("# roofline: 16.04 GB streamed / 900 GB/s = 17.8 ms/token = 56.1 tok/s")
        print("#   (trunk 15.24 + lm_head 0.80; embed_tokens and the visual tower are")
        print("#    resident but not streamed — errors/2026-09-02-roofline-is-the-streamed-subset)")
        # A profile that does not roughly reconcile with the clock is measuring the
        # wrong window — profiling across the prefill chunks once read 8217 ms/token
        # against a 33 ms token.
        if gpu_ms > 2 * wall_ms:
            print(f"\n!! {gpu_ms:.0f} ms GPU inside a {wall_ms:.0f} ms forward — the "
                  "window includes work that is not this tick's. Numbers unusable.")
        print()
        print(f"{'class':>14} {'ms/fwd':>8} {'% GPU':>7}")
        for cls, us in sorted(by_cls.items(), key=lambda kv: -kv[1]):
            print(f"{cls:>14} {us/1000/max(fwd,1):>8.2f} {100*us/total:>6.1f}%")

        print(f"\n{'kernel':>52} {'ms/fwd':>8} {'calls/fwd':>10}")
        for name, (us, cnt) in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:20]:
            print(f"{name[-52:]:>52} {us/1000/max(fwd,1):>8.2f} {cnt/max(fwd,1):>10.1f}")

        # No per-shape GEMV table here, and that is a measured dead end rather than an
        # omission: `record_shapes=True` groups by the ATen input shapes of the op that
        # launched the kernel, and a TileLang kernel arrives as a raw CUDA kernel name
        # with no ATen op above it -- so all 330 launches came back under one empty
        # shape row (measured, bud7). The rung sweep below is the working instrument
        # for what M costs.

    # The whole point of >1 ctx in one process: which CLASS carries the step. A
    # class that is flat across contexts cannot be the cause however large it is.
    if len(rows) > 1:
        lo, hi = ctxs[0], ctxs[-1]
        a, b = rows[lo][0], rows[hi][0]
        fa, fb = rows[lo][3], rows[hi][3]
        print(f"\n# what changed, ctx {lo} -> {hi}, per FORWARD")
        print(comparable(rows[lo][5], rows[hi][5]))
        print(f"{'class':>14} {'ms lo':>8} {'ms hi':>8} {'delta':>8} {'share':>7}")
        deltas = {c: b.get(c, 0) / 1000 / max(fb, 1) - a.get(c, 0) / 1000 / max(fa, 1)
                  for c in set(a) | set(b)}
        grew = sum(d for d in deltas.values() if d > 0)
        for c, d in sorted(deltas.items(), key=lambda kv: -kv[1]):
            print(f"{c:>14} {a.get(c,0)/1000/max(fa,1):>8.2f} "
                  f"{b.get(c,0)/1000/max(fb,1):>8.2f} {d:>+8.2f} "
                  f"{100*d/grew if d > 0 and grew else 0:>6.1f}%")

    # The rung step, by CLASS, at one context. A verify at rung W re-streams the
    # same weights as W=1 and launches the same kernels the same number of times
    # -- only M grows -- so whatever carries the 5.68/8.06/8.71 ms per extra row
    # has to show up here as a class that grows while the launch count does not.
    # This is the number the block-parallel reject rests on: rung 8's 80.31 ms
    # against our whole k=3 tick's 62.74.
    if args.depths:
        if draft is None:
            raise SystemExit("--depths needs --draft: without a draft every tick is rung 1")
        depths = [int(d) for d in args.depths.split(",")]
        ctx = ctxs[0]
        drows = {}
        for d in depths:
            draft.set_depth(d)
            e._width = draft.width
            assert e._width == d + 1, f"engine kept width {e._width} for depth {d}"
            drain(to_decode(ctx, args.tokens))  # warm THIS width's capture
            torch.cuda.synchronize()
            drows[d] = profile_one(ctx)
        print(f"\n# the rung step at ctx={ctx}, per FORWARD. Same weights, same launch "
              "count at every depth; only M moves.")
        hdr = "".join(f"{f'd{d}':>9}" for d in depths)
        print(f"{'class':>14}{hdr}")
        classes = {c for d in depths for c in drows[d][0]}
        for c in sorted(classes, key=lambda c: -drows[depths[0]][0].get(c, 0)):
            cells = "".join(f"{drows[d][0].get(c, 0)/1000/max(drows[d][3],1):>9.2f}"
                            for d in depths)
            print(f"{c:>14}{cells}")
        print(f"{'-- GPU total':>14}"
              + "".join(f"{sum(drows[d][0].values())/1000/max(drows[d][3],1):>9.2f}"
                        for d in depths))
        print(f"{'-- rung':>14}"
              + "".join(f"{next(w for w in LADDER_WIDTHS if w >= round(drows[d][5])):>9d}"
                        for d in depths))
        # The control: if the launch count moves, the depths are not running the same
        # kernels and a per-class delta is not an M cost.
        def gemv_calls(d):
            by_k = drows[d][1]
            n = sum(c for k, (_, c) in by_k.items() if classify(k) == "fp4 GEMV")
            return n / max(drows[d][3], 1)

        print(f"{'-- GEMV calls':>14}" + "".join(f"{gemv_calls(d):>9.1f}" for d in depths))


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
