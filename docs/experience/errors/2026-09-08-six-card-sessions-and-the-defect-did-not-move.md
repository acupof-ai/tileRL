# Six card sessions on the 2.6x rollout tick, and the defect is where it was — 2026-09-08

**Status:** the arm is abandoned, not paused. The 2.643x gap
([the entry](2026-09-08-the-training-rollout-tick-is-2.6x-serving.md)) is still ~100%
unattributed. No seventh session: ckl's ruling put the P1 gate on the critical path, and six
sessions of mine produced no attribution.

## The one sentence that explains all six

**A two-engine-in-one-process arm introduces a confound the recorded measurement does not
have, so a positive result would have been uninterpretable and the negative results were the
confound.** (`tilerl-27`'s wording, which is exactly right and is why it is quoted rather
than paraphrased.)

The recorded 55.21 / 61.68 / 23.34 ms/tick came from **three separate processes**. My arm
built both engines in one process so that config would be the only variable. It succeeded at
that and thereby introduced a variable no config field can express: **build order**.

**The rule:** match the instrument's process topology to the measurement you are comparing
against, *before* matching its parameters. I aligned pool, batch, context and slots — four
fields — while the variable that actually moved the number was not a field at all.

## What each session cost

| # | tag | what it produced | why it did not answer the question |
|---|---|---|---|
| 1 | `rollgap` | nothing | anchor rejected itself at ratio **1.00068** against a `<= 1.0` bound |
| 2 | `rollgap2` | `tok/fwd 1.00` | B=1, not B=8 — a different quantity |
| 3 | `rollgap3` | term2 **2.180x** | pools differed 8.8x (232 vs 2048 blocks) — confounded |
| 4 | `matched` | nothing | anchor failed at ratio **1.027**, broken by my own session-1 fix |
| 5 | `matched2` | term2 **2.157x** at equal pools | pool killed as the mechanism; order not yet suspected |
| 6 | `order` | serve **2.162x** slower built second; train **1.006x** either way | the 2.16x in arms 3 and 5 was build order |

Three sessions lost to my own instrument (1, 2, 4), one to a hypothesis I killed with my own
control (3→5), two to the confound (5, 6).

## The numbers that survive

All full-27B, `--tokens 128`, `--group 8`, graph on, one card, both arms in one process.
`ms/tok` is wall; `devms/fwd` is CUDA events around `step()`; `resid` is `wall − Σdevice`.

| session | arm | blocks | ms/tok | fwd/tok | devms/fwd | resid/tok | tok/fwd |
|---|---|---:|---:|---:|---:|---:|---:|
| 3 `rollgap3` | train | 232 | 3.34 | 0.128 | 25.91 | 0.01 | 7.79 |
| 3 `rollgap3` | serve | 2048 | 7.26 | 0.128 | 56.49 | 0.01 | 7.79 |
| 5 `matched2` | train (1st) | 1096 | 3.36 | 0.128 | 26.10 | 0.01 | 7.79 |
| 5 `matched2` | serve (2nd) | 1096 | 7.24 | 0.128 | 56.29 | 0.01 | 7.79 |
| 6 `order` | serve (1st) | 1096 | 3.35 | 0.128 | 26.04 | 0.01 | 7.79 |
| 6 `order` | train (2nd) | 1096 | 3.34 | 0.128 | 25.95 | 0.01 | 7.79 |

Three findings are real and independent of the confound:

**1. The residual is not where the gap lives — the prediction is refuted four times over.**
[The prediction on record](../wins/2026-09-08-prediction-for-the-rollout-tick-arm.md) said
the residual carries the majority of the 2.279x. Measured: **0.009–0.010 ms/tok in every
arm, 0.1–0.3% of the tick.** Term 1 is 1.000x between arms (`tok/fwd` 7.79 both sides, so
the partly-drained batch of the 09-06 run is not reproduced here either). Whatever the arm
was measuring, it was measuring it entirely in device time.

**2. Pool geometry is dead as the mechanism.** Session 3's 2.180x came with an 8.8x pool
difference and I fitted an exponent through it, claiming 23.8% of the per-forward remainder.
Session 5 equalized the pool at 1096 blocks and term2 read 2.157x — **8.8x of pool variation
moved term2 by 1.1%.** The 23.8% claim is withdrawn entirely, not revised: it was fitted
through two points that differed in pool size *and* everything else.

**3. Build order is asymmetric, and nobody knew this yesterday.** `serve` built second is
**2.162x** slower than `serve` built first (56.29 vs 26.04). `train` is position-invariant:
**1.006x** (25.95 second vs 26.10 first). So order is necessary but not sufficient — it
degrades one config and not the other, which is a property of the two configs and not of the
harness alone. It is also the whole of the 2.16x that sessions 3 and 5 reported.

## The instrument failures, each with its mechanism

**Session 1 — a bound with no tolerance, over an interval that measured a compile.** The
anchor read 10555.6 ms device against 10548.4 ms wall. Two errors, and the second is the
expensive one: event slop needs a tolerance (7.2 ms here), *and* a cold engine spends 10.5 s
of the interval in four TileLang compiles. Fixing only the tolerance would have made the
anchor **pass while bounding nothing** — the JIT dominates both sides and the comparison
becomes vacuous. Fixed by warming before timing.

**Session 2 — the shape was never asserted, only printed.** The arm took one prompt and
reported `tok/fwd 1.00`. B=1 is a different quantity from the B=8 tick under investigation,
and the table printed the evidence without gating on it. Now `decode_terms` takes a list and
`main` refuses unless `tok_per_fwd >= group - 1` and `usable_slots >= group` before timing
starts.

**Session 4 — my session-1 fix broke my session-1 bound.** Warming shortened the anchor
interval **67x**, and my tolerance was a *ratio* while event slop is *absolute* (4.3 ms:
0.07% of the cold interval, 2.7% of the warm one). Replaced with
`dev > wall + max(10.0, wall*0.02)`, sized against the ±157 ms failure modes it exists to
catch. **A tolerance calibrated on one interval length is not a tolerance on the quantity.**

**All four runs — LoRA was attached to the wrong arm.** The attach was gated
`if name == "train"` at the *bottom* of the config loop, so train ran without the adapter and
serve ran with it — the reverse of both real configs. Now attached to neither, and LoRA is
its own arm whenever one is run.

**All six runs — the closure gate could not fail.** Every arm printed `closes: yes` and it
was evidence of nothing. Expand it:

    fwd_per_tok × dev_ms_per_fwd = (fwd/tok)(dev_ms/fwd) = dev_ms/tok
    resid_ms_per_tok             = (wall_ms − dev_ms)/tok
    sum                          = wall_ms/tok             identically

`fwd` cancels, so a 100x-wrong forward count leaves it exact; term 3 is *defined* as the
remainder, so no fourth term can exist. Measured: a 10x-wrong device time and a 100x-wrong
forward count both pass at err = 0.00e+00. Found by `tilerl-27`, who wrote the gate and then
found the hole in it. Replaced by `dev_under_wall` — `dev_ms <= wall_ms`, an inequality that
can actually be violated.

## Known limitation of the surviving anchor

The anchor times **eager prefill ticks** while the conclusion rests on **replayed decode
graphs** — a different tick type on a different execution path, and precisely where a CUDA
event might attach differently. It is the best outside bound available (`_prefill_secs` is
the only wall figure the engine already keeps) but it is adjacent to what term 2 measures,
not identical to it. Stated here because nothing in the arm would notice. (`tilerl-27`.)

The lower bound has the same shape of gap: `ratio < 0.05` catches events that measure
nothing, and misses events that measure *part* of the window. A floor set from a known-good
run would catch that; none has been recorded, so it is not set.

## Rules

- **Match the instrument's process topology to the measurement you compare against, before
  matching its parameters.** Aligning fields is the visible work; the topology is the
  invisible variable, and no config field can express it.
- **A self-consistency gate cannot detect a wrong input.** Closure and the anchor check the
  *arithmetic* of a decomposition, so they pass any self-consistent run — including one of
  the wrong workload. Every arm needs at least one assertion about the **shape of the
  workload**: batch, context, graph on/off, W.
- **A wall-clock interval bounds device time only if something forces completion before it
  closes.** Otherwise it is not a bound in either direction.
- **A tolerance calibrated on one interval length is not a tolerance on the quantity.** Fix
  the tolerance and the interval together, or the next fix breaks the bound.
- **A fix to a bound is a change to the thing bounded.** Session 1's warm fix was correct and
  it invalidated session 1's tolerance.

## What is next, and what is not

The correct next arm is **two processes** — one engine per process, same card, same sha, each
claiming through `pod_run.sh` — which reproduces the recorded topology. It is on the board
behind the P1 gate; whether to spend a card on it is ckl's call, not mine.

The ledger defect this exposed is separate and does not need a card: the run manifest
(`cli.py:494-509`) records recipe/commit/data/seed/lr/lora_rank but none of `num_blocks`,
`max_total_tokens`, `num_slots`, `decode_graph` — the exact bundle this arm exists to
isolate. Both pools tonight were recoverable **only because the probe script logged its own
flags**; the manifest never helped. Since per-forward device time moves with config, a record
that omits the config cannot support a wall-clock comparison between two runs — and P5
(against verl+sglang) is entirely that kind of comparison.
