"""The 2.6x training-rollout gap: which term of the identity carries it.

Two of our own full-27B B=8 decode measurements disagree by 2.64x with the slower one on
the SHORTER context -- 55.21 and 61.68 ms/tick training against 23.34 serving, all
graph-on, all `wall / decode_forwards`
(errors/2026-09-08-the-training-rollout-tick-is-2.6x-serving.md). Worth 45.9% of a GRPO
step, 93% unattributed.

`ms/tick` cannot locate it: it is `wall / decode_forwards`, and the denominator can itself
differ between the arms. This measures the identity instead, per arm:

    wall/token = (forwards/token) x (device ms/forward) + (residual ms/token)

The prediction is on record BEFORE this ran, in
wins/2026-09-08-prediction-for-the-rollout-tick-arm.md: term 1 is 1.159x of the 2.643x
(tok/fwd 6.90 vs a nominal 8), so 2.279x has to sit in device time or the residual, and I
expect the RESIDUAL to carry the majority -- both arms replay the same captured graph at
the same shapes, so a 2.3x device difference needs a different kernel or shape and the arms
agree on both. Term 2 differing by more than ~1.3x refutes that.

## Two gates, because one of them cannot fail

CLOSURE is necessary and insufficient. Term 3 is DERIVED as `wall - sum(device)`, so
`wall = device + residual` holds by construction for any term 2 whatsoever -- a CUDA-event
measurement wrong by 10x passes, with terms 2 and 3 absorbing the error in equal and
opposite amounts. Closure tests term 1 against the total and nothing else.

So term 2 is bounded from OUTSIDE the identity, against `_prefill_secs` (`engine.py:483`,
accumulated `:1023`), which the engine already publishes. Two properties of that anchor,
read rather than assumed:

  * It is `time.perf_counter()`, so WALL not device -- an upper bound, and the slack
    between the two IS the host overhead being measured elsewhere. Reading it as an
    equality would assume away the answer.
  * On a prefill-only tick nothing inside the interval forces completion.
    `_sample_commit` (`:1018`) is in the `else` of `if chains`, not under `if decodes`, so
    it runs with an empty list and `_sample_batch` returns `[]` at `:1309-1310` before any
    `.tolist()`. `_finish_prefills` is at `:1024`, after the accumulation. So the interval
    can close with kernels in flight and `_prefill_secs` can be SMALLER than the device
    time it contains -- the bound then fails on a correct instrument, in the direction that
    looks like a finding.

Hence the synchronize inside `anchor()`: forced completion before the interval closes, in
that loop only. Patching `Engine.step` globally would also sync every decode tick and
corrupt the measurement the anchor exists to protect. The rule that generalises: a
wall-clock interval bounds device time only if something forces completion before it
closes.

## What a positive result does and does not say

The two arms differ in a BUNDLE of engine config, not one field. A reproduction localizes
to the bundle and names no mechanism. The bisection is pre-committed to three groups (pool
geometry / slot and batch shape / store and sampling) so a positive result is one of three
follow-ups, not an eleven-way search.

Usage (on the pod, card claimed per python pid by pod_run.sh):

    scripts/pod_run.sh --wait rollgap 6 -- python3 scripts/prof_rollout_gap.py \
        --source /data00/Qwen3.8-27B-NVFP4 --tokens 128
"""

from __future__ import annotations

import argparse
import os
import time


def _sync():
    """No-op without CUDA. The arm is card-only, but its GATES must be testable without one:
    a precondition that has only ever run on the correct path proves nothing, and the shape
    gate is what would have caught the B=1 run. Off-card, term 2 reads 0 -- meaningless, but
    the token and forward counters are not, and those are what the gate reads.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _events(n):
    import torch

    if not torch.cuda.is_available():
        return [(None, None)] * n
    return [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(n)]


def decode_terms(e, batch, n, sp, mod, cap=4096):
    """The three terms over DECODE ticks only, prefill burned first.

    `batch` is a LIST of prompts, submitted together and driven to completion together --
    the quantity under test is a B=8 batched tick, and a single request measures a
    different thing. The first version of this took one prompt and produced a table
    reading `tok/fwd 1.00`, which is B=1; `main` now asserts the shape rather than
    printing it.

    Per-tick CUDA events around `step()`, so term 2 is device time and term 3 is what is
    left of the wall. `cap` bounds the event list: an unbounded one on a runaway loop is
    how a probe eats the host.
    """

    rids = [e.submit(ids, sp(n)) for ids in batch]
    # Every row must be in DECODE before timing starts, or the window charges some rows'
    # prefill into the decode figure.
    while True:
        reqs = [next((r for r in e._running if r.req_id == i), None) for i in rids]
        if all(r is not None and r.phase == mod._PHASE_DECODE for r in reqs):
            break
        if any(r is None for r in reqs) and any(e.take(i) is not None for i in rids):
            raise SystemExit("a request finished during prefill -- prompt too short for n")
        e.step()

    _sync()
    s0, t0 = e.stats(), time.perf_counter()
    evs, out, i = _events(cap), {}, 0
    while len(out) < len(rids):
        if i >= cap:
            raise SystemExit(f"decode exceeded {cap} ticks: the loop is not converging")
        a, b = evs[i]
        if a is not None:
            a.record()
        e.step()
        if b is not None:
            b.record()
        i += 1
        for rid in rids:
            if rid not in out:
                r = e.take(rid)
                if r is not None:
                    out[rid] = r
    _sync()
    wall, s1 = time.perf_counter() - t0, e.stats()

    # 0.0 without CUDA: term 2 is meaningless on CPU, but `tok` and `fwd` are not, and the
    # shape gate reads those. So the arm's preconditions are testable without a card.
    dev_ms = sum(a.elapsed_time(b) for a, b in evs[:i] if a is not None)
    tok = max(s1["tokens_generated"] - s0["tokens_generated"], 1)
    fwd = max(s1["decode_forwards"] - s0["decode_forwards"], 1)
    return {
        "wall_ms_per_tok": wall * 1000 / tok,
        "fwd_per_tok": fwd / tok,
        "dev_ms_per_fwd": dev_ms / fwd,
        "resid_ms_per_tok": (wall * 1000 - dev_ms) / tok,
        "ticks": i, "tok": tok, "fwd": fwd, "tok_per_fwd": tok / fwd,
        "dev_ms": dev_ms, "wall_ms": wall * 1000,
    }


def dev_under_wall(t):
    """`dev_ms <= wall_ms`: a real inequality, unlike the closure check it replaces.

    The first version of this module gated on "term1 x term2 + term3 rebuilds
    wall/token" and claimed that caught a wrong term 1 or a fourth unnamed term. It caught
    neither, and it could not fail at all -- expand it:

        fwd_per_tok * dev_ms_per_fwd = (fwd/tok)(dev_ms/fwd) = dev_ms/tok
        resid_ms_per_tok             = (wall_ms - dev_ms)/tok
        sum                          = wall_ms/tok            identically

    `fwd` cancels, so a 100x-wrong forward count leaves it exact; term 3 is DEFINED as the
    remainder, so no fourth term can exist. Measured: a 10x-wrong device time and a
    100x-wrong forward count both pass at err = 0.00e+00. Six arms reported "closes: yes"
    as evidence and it was evidence of nothing. (Found by tilerl-27.)

    Device time inside a wall interval genuinely cannot exceed it, so this can fail.
    """
    return t["dev_ms"] <= t["wall_ms"], t["dev_ms"] / t["wall_ms"]


def anchor(e, sp, mod, ids, warm=True):
    """Bound the CUDA events from OUTSIDE the identity, against the engine's own
    `_prefill_secs`.

    LIMITATION, stated because nothing in the arm would notice it: this certifies the CUDA
    events on EAGER PREFILL ticks and the conclusion rests on them measuring REPLAYED DECODE
    GRAPHS -- a different tick type and a different execution path, and precisely where an
    event might attach differently. The anchor is the best outside bound available (it is the
    only wall figure the engine already keeps) but it is adjacent to what term 2 measures,
    not identical to it. (tilerl-27.)

    Times prefill ticks with the same events, against the engine's own wall accumulation
    for those ticks. `_prefill_secs` is not in `stats()` (only `prefill_rate` is, at
    `engine.py:864`), so the private attribute is read directly -- this is a diagnostic,
    and deriving it from the rate would divide by a token count that includes every
    earlier tick.

    Returns (device_ms, wall_ms, ratio). Two things the first version of this got wrong,
    both found by running it:

      * **The bound is `ratio <= 1` plus event error, not `<= 1.0`.** Measured 10555.6 ms
        device against 10548.4 ms wall -- ratio 1.00068, and a bare `<= 1.0` called the
        instrument broken. The events bracket the wall interval (recorded outside the
        loop the accumulation runs in), so the device figure legitimately exceeds it by
        the record-to-record overhead. 2% covers it.
      * **A cold engine measures the JIT, not the forward.** That 10.5 s is four TileLang
        compiles (`write_tokens`, `paged_attention`, `paged_attention_decode`), not a
        prefill. The ratio was ~1.0 because both clocks were timing the same compile, so
        the anchor would have *passed* while bounding nothing. Hence `warm=True`: one
        short request first, so every kernel is compiled before the interval opens.
    """

    if warm:  # a compile inside the interval makes both clocks time the JIT, not the forward
        rid = e.submit(ids, sp(2))
        while e.take(rid) is None:
            e.step()
        _sync()

    w0, s0 = e._prefill_secs, e._prefill_tokens
    (a, b), = _events(1)
    rid = e.submit(ids, sp(4))
    _sync()
    if a is not None:
        a.record()
    # Synchronize INSIDE this loop only: the wall figure `_prefill_secs` accumulates must
    # not close with kernels in flight, or it bounds nothing. Patching `Engine.step`
    # globally would also sync every DECODE tick and corrupt the measurement this anchor
    # exists to protect, so the sync is here and nowhere else.
    while e._prefill_tokens == s0:  # exactly the ticks that accumulated into _prefill_secs
        e.step()
        _sync()
    if b is not None:
        b.record()
    _sync()
    dev_ms = a.elapsed_time(b) if a is not None else 0.0
    wall_ms = (e._prefill_secs - w0) * 1000
    while e.take(rid) is None:  # drain, so the next arm starts on an empty engine
        e.step()
    if a is None:  # no CUDA: nothing to bound, and the caller must not read this as a pass
        return 0.0, wall_ms, None
    return dev_ms, wall_ms, dev_ms / wall_ms if wall_ms > 0 else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompt", type=int, default=256, help="rollout's prompt length")
    ap.add_argument("--group", type=int, default=8, help="B, the rollout's group size")
    ap.add_argument("--train-blocks", type=int, default=0,
                    help="override the train arm's num_blocks (0 = cli.py:543's formula)")
    ap.add_argument("--serve-blocks", type=int, default=2048,
                    help="the serve arm's num_blocks; 1096 reproduces the recorded 23.34 arm")
    ap.add_argument("--serve-first", action="store_true",
                    help="build the serve arm first: the control for build order, which has "
                         "been confounded with config in every arm run so far")
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    os.environ.setdefault("TILERL_QWEN38_SOURCE", args.source)

    import torch
    from tilerl_kernels.backend import get_backend

    from tilerl import cli
    from tilerl import engine as mod
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore

    backend = get_backend()

    def sp(n):
        return SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=n, seed=0)

    # ONE process, ONE card, ONE set of weights: config is the only variable. Two models
    # would make weight layout a second difference and the arm would name nothing.
    cfg, model = cli._build_model("qwen38-27b", seed=0, fuse_projections=False)
    ctx = args.prompt + args.tokens + 64
    prompts = [list(range(1000 * (i + 1), 1000 * (i + 1) + args.prompt))
               for i in range(args.group * 2 + 1)]

    # Arm A -- the training rollout's engine, copied field for field from cli.py:543-547.
    # Arm B -- the serving config: cli._build_engine's own defaults, which is what the
    # 23.34 ms/tick arm ran. The difference between the two IS the bundle under test.
    def build_train():
        # Field for field from cli.py:543-547, unmodified: `build_engine` sizes the state
        # pool as `num_slots + pad` (`engine.py:1616`), so the shipped num_slots=8 yields 8
        # usable slots and holds B=8. Verified on CPU rather than reasoned about -- I first
        # "corrected" this to group+1 believing usable_slots subtracted the pad row from
        # the request, which would have changed a field of the bundle under test for no
        # reason.
        return build_engine(
            cfg, model, backend, num_slots=args.group, max_batch=args.group,
            num_blocks=args.train_blocks or -(-ctx // BLOCK_TOKENS) * 8 + 8,
            max_total_tokens=max(ctx, 8192),
            decode_graph=True, prefix_store=NoPrefixStore(),
        )

    def build_serve():
        # `slots=args.group` and not cli's shipped `slots=4`: at 4 the excess rows queue
        # (`engine.py:673-681`) and the run completes at half the concurrency, which is what
        # the previous session's B=1 table was. The recorded 23.34 serving figure was itself
        # B=8 -- `--prompts 8 --batch 8`, quoted in
        # wins/2026-09-06-b8-speculation-loses-to-no-speculation.md:79 -- so slots is set to
        # admit that batch in both arms; it moves from the bundle to the controlled set and
        # the bisection's slot+batch group is smaller by one.
        return cli._build_engine(cfg, model, backend, slots=args.group,
                                 blocks=args.serve_blocks, max_ctx=4096,
                                 max_batch=args.group)

    # `blocks` is a printed column, not a footnote: this arm's whole finding is that
    # per-forward device time moves with pool size, so a table without it cannot be read.
    # The run manifest records recipe/commit/seed/lr but none of num_blocks,
    # max_total_tokens, num_slots or decode_graph -- which is why both recorded arms'
    # pools were recoverable only from probe scripts that logged their own flags.
    print(f"{'arm':>8} {'blocks':>7} {'ms/tok':>8} {'fwd/tok':>8} {'devms/fwd':>10} "
          f"{'resid/tok':>10} {'tok/fwd':>8} {'anchor':>7} {'dev/wall':>8}")
    rows = {}
    # Build ORDER, not just build config. Whichever engine is built second inherits a
    # fragmented caching allocator, a different graph pool and a warm JIT cache -- and
    # train was built first in all four arms run so far, so order has been confounded with
    # config throughout. This flag is the control: if term2 inverts when the order does,
    # the effect is ordering and the two-engine design is the instrument rather than the
    # system. It is also the only hypothesis on the table that predicts BOTH these arms'
    # direction and the recorded pair's opposite sign, since the recorded arms were separate
    # processes where no ordering effect exists. (tilerl-27's.)
    arms = [("train", build_train), ("serve", build_serve)]
    if args.serve_first:
        arms.reverse()
    # LoRA is attached to NEITHER arm, and that is a change from the first four runs. It used
    # to be attached after the train arm was measured and before the serve arm, so train ran
    # WITHOUT the adapter and serve ran WITH it -- the reverse of both real configs, and
    # inverted again by --serve-first. Priced at <=4% of the gap either way, so the honest
    # move is to hold it out of the comparison entirely rather than apply it backwards.
    # It belongs in its own arm, one variable at a time.
    for name, build in arms:
        e = build()
        # Before any timing: can this engine even HOLD the batch under test? A slot is held
        # from submit to finish (`engine.py:673-681`), and a shortfall neither raises nor
        # drops -- `submit` has no slot check and `_admit` returns False ("does not fit
        # yet"), so the excess rows QUEUE. Measured by v100 at usable_slots=4, max_batch=8:
        # 8 submits all succeed and the batch serializes into two halves of 4, producing a
        # table with twice the ticks and half the rows per tick. That is the worst of the
        # three failure modes -- a raise cannot be misread as data, a drop shows in the
        # counters, a queued run looks normal. So this gate exists to stop a run that
        # SUCCEEDS while measuring the wrong concurrency, not to pre-empt a crash.
        if e.usable_slots < args.group:
            raise SystemExit(
                f"{name}: {e.usable_slots} usable slots against B={args.group} -- the excess "
                f"rows would queue rather than raise, and the table would read as a normal "
                f"run at half the concurrency. Pass num_slots={args.group}: `build_engine` "
                f"adds the decode graph's pad row itself (`engine.py:1616`, num_slots + pad), "
                f"so B is sufficient even with the graph on -- the engine's own warning says "
                f"'+ 1 for the decode graph's pad row' and that is false (v100, #296)."
            )
        # The anchor first: if the events cannot bound a prefill tick, nothing below means
        # anything, and it costs one short request to find out.
        dev, wall, ratio = anchor(e, sp, mod, prompts[-1])
        # 1.02, not 1.0: the events are recorded outside the loop the accumulation runs in,
        # so the device figure legitimately exceeds the wall by the record-to-record
        # overhead. Measured 1.00068 on a warm engine. The 0.05 floor catches events that
        # are not spanning the forward at all.
        # The bound is absolute-plus-relative, not a pure ratio, and that took two runs to
        # learn. The events bracket the wall interval (recorded outside the loop that
        # accumulates it), so the device figure exceeds the wall by a FIXED cost -- measured
        # 7.2 ms on a cold 10548 ms interval (ratio 1.00068, passed) and 4.3 ms on a warm
        # 157.6 ms one (ratio 1.027, failed a 1.02 ratio bound). Same slop, 39x the
        # fraction, because the warm fix shortened the interval 67x. A tolerance calibrated
        # on one interval length is not a tolerance on the quantity.
        # 10 ms, sized against what it must DETECT rather than against the interval: the
        # failures worth catching are events that do not span the forward (excess ~ -wall)
        # and events that double-count (excess ~ +wall), both ~157 ms here. 10 ms admits the
        # 4.3-7.2 ms observed and still rejects either by 15x. A 20 ms allowance would be
        # 12.7% of a warm interval, which is loose enough to hide a real error.
        slop_ms, slop_frac = 10.0, 0.02
        if ratio is None:
            # No CUDA: term 2 is unmeasurable, so the arm must not print a decomposition.
            # The shape gate below still runs, which is the point of the CPU path.
            print(f"{name}: no CUDA -- term 2 unmeasurable, gates only")
        elif dev > wall + max(slop_ms, wall * slop_frac) or ratio < 0.05:
            raise SystemExit(
                f"{name}: CUDA events read {dev:.1f} ms device against a {wall:.1f} ms wall "
                f"interval containing them (ratio {ratio:.3f}, excess {dev - wall:+.1f} ms "
                f"against a {max(slop_ms, wall * slop_frac):.1f} ms allowance). The "
                "instrument is wrong, and closure below cannot detect that -- term 3 is "
                "derived from the wall and absorbs any error in term 2."
            )
        decode_terms(e, prompts[:args.group], 8, sp, mod)  # warm: capture every (B, W)
        t = decode_terms(e, prompts[args.group:args.group * 2], args.tokens, sp, mod)
        # The shape is a PRECONDITION, not a column to read past. The first run of this
        # printed tok/fwd 1.00 -- a B=1 decode -- in a table whose every other number
        # looked plausible, and the anchor and closure gates both passed, because they
        # check the decomposition's arithmetic and not whether the workload is the one
        # under test.
        if t["tok_per_fwd"] < args.group - 1:
            raise SystemExit(
                f"{name}: tok/fwd {t['tok_per_fwd']:.2f} against a requested B={args.group}. "
                f"The 2.6x under test is a batched-decode figure, so a partly-drained or "
                f"single-row batch measures a different quantity. slots_used="
                f"{e.stats().get('slots_used')} slots_total={e.stats().get('slots_total')}"
            )
        ok, dev_frac = dev_under_wall(t)
        t["anchor"] = ratio
        rows[name] = t
        t["blocks"] = e.usable_blocks  # the engine's own answer, not the requested figure
        print(f"{name:>8} {t['blocks']:>7} {t['wall_ms_per_tok']:>8.2f} "
              f"{t['fwd_per_tok']:>8.3f} "
              f"{t['dev_ms_per_fwd']:>10.2f} {t['resid_ms_per_tok']:>10.2f} "
              f"{t['tok_per_fwd']:>8.2f} "
              f"{'n/a' if ratio is None else f'{ratio:.2f}':>7} "
              f"{dev_frac:>7.3f}")
        if not ok:
            raise SystemExit(
                f"{name}: device time {t['dev_ms']:.1f} ms exceeds the {t['wall_ms']:.1f} ms "
                f"wall interval containing it ({dev_frac:.3f}x). The events are not measuring "
                "work inside the window they were recorded around."
            )
        e.shutdown()
        del e
        torch.cuda.empty_cache()

    a, b = rows["train"], rows["serve"]
    gap = a["wall_ms_per_tok"] / b["wall_ms_per_tok"]
    t1 = a["fwd_per_tok"] / b["fwd_per_tok"]
    t2 = a["dev_ms_per_fwd"] / b["dev_ms_per_fwd"]
    t3 = ((a["resid_ms_per_tok"] / b["resid_ms_per_tok"])
          if b["resid_ms_per_tok"] > 0 else float("inf"))
    print(f"\ngap {gap:.3f}x   term1 {t1:.3f}x   term2 {t2:.3f}x   term3 {t3:.3f}x")
    print(f"resid {a['resid_ms_per_tok']:.3f} vs {b['resid_ms_per_tok']:.3f} ms/tok "
          f"({a['resid_ms_per_tok'] / a['wall_ms_per_tok']:.1%} and "
          f"{b['resid_ms_per_tok'] / b['wall_ms_per_tok']:.1%} of the tick)")
    # Two-sided, and it reports the direction rather than grading itself. The first version
    # tested `t2 > 1.3 -> refuted, else consistent`, which read a term2 of 0.514x -- a 2x
    # difference the OTHER way -- as confirmation of a prediction about a 2.279x excess.
    if gap <= 1.05:
        print(f"\nNO GAP REPRODUCED (train/serve = {gap:.3f}x). The recorded defect is "
              f"2.643x with training SLOWER; this arm has training "
              f"{'faster' if gap < 1 else 'level'}. Nothing about the prediction is tested: "
              f"there is no excess for any term to carry. Either the bundle is not the "
              f"cause, or this arm does not reproduce the recorded conditions.")
    else:
        share1, share2 = (t1 - 1) / (gap - 1), (t2 - 1) / (gap - 1)
        # Shares of the EXCESS, not of the total: a ratio on a part read as a ratio on the
        # whole is the denominator defect this project has paid for three times.
        print(f"\nterm1 explains {share1:.1%} of the excess, term2 {share2:.1%}, "
              f"residual {1 - share1 - share2:.1%}")
        print("Prediction (068bc74, before any of this ran): term1 ~1.159x of 2.643x, the "
              "RESIDUAL carries the majority, REFUTED by term2 > ~1.3x.")
        if t2 > 1.3:
            print(f"REFUTED: term2 {t2:.3f}x. The replay itself is slower -- pool geometry, "
                  "not python.")
        elif 1 - share1 - share2 > 0.5:
            print(f"HELD: term2 {t2:.3f}x and the residual carries "
                  f"{1 - share1 - share2:.1%}. Host-side.")
        else:
            print(f"NEITHER: term2 {t2:.3f}x is under the refutation threshold but the "
                  f"residual carries only {1 - share1 - share2:.1%}. The gap is spread, "
                  "which the prediction did not anticipate.")
    print("A reproduction localizes to the CONFIG BUNDLE, not a mechanism. Next: three "
          "pre-committed groups (pool geometry / slot+batch shape / store+sampling).")


if __name__ == "__main__":
    main()
