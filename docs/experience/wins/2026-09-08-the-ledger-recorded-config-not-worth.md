# The ledger recorded how a run was configured and not what it was worth — 2026-09-08

**Status:** fixed. The run manifest now carries the engine config bundle and a
`(step, score, cumulative_secs)` curve, and `tilerl ledger --time-to-score` reads the
objective back off it. None of the three existed before, and
`time_to_score = steps_to_score × seconds_per_step` needs all of them.

## Context

ckl set the project objective as `time_to_score = steps_to_score × seconds_per_step`. We
have measured the second factor repeatedly — 85.617 s per GRPO step
([the step is 74% rollout](2026-09-07-the-step-is-74-percent-rollout.md)) — and the manifest
records enough to identify a run: recipe, commit, data hash, seed, lr, lora_rank, tp,
reward, eval settings (`cli.py:517-535`).

It records neither factor's *context*. Two gaps, found from opposite directions.

## Gap 1: the engine config, found by needing it and not having it

Six card sessions on the 2.6x rollout tick
([the entry](../errors/2026-09-08-six-card-sessions-and-the-defect-did-not-move.md)) measured
per-forward device time moving **2.16x** with engine configuration. The manifest records
none of `num_blocks`, `max_total_tokens`, `num_slots`, `decode_graph` — exactly the bundle
the 2.6x defect names as where the gap must live.

Both pool sizes in that arm were recoverable **only because the probe script logged its own
flags**. The manifest never helped, across four arms.

So: a record that omits the engine config cannot support a wall-clock comparison between two
runs. P5 — against verl+sglang — is entirely that kind of comparison.

**Where it goes matters as much as that it goes somewhere.** `id = run_id(inputs)`
(`ledger.py:41`), so in this ledger `inputs` **is** the id: there is no record-only slot
inside it. A pool recorded there makes every pool change a different run and defeats the
idempotence a rerun relies on. The rule that settles it: **comparability should not change
the id.** New top-level key `manifest["engine"]`, beside `metrics` / `gates` / `artifacts`.

**Read off the built engine, not the call's kwargs.** `max_blocks` clamps `num_blocks`, and
the captured tick reserves a pad row in both pools, so the argument and the pool disagree —
`Engine.config` reports `usable_blocks` / `usable_slots`. Recorded a second time at `_finish`
because `Engine._graph_for` sets `_decode_graph_on = False` in its `except` on a capture
failure: a build-time snapshot can otherwise claim graph-on for a run that decoded eagerly,
which is the stale-record failure the key exists to prevent. (Named by function, not line:
that assignment moved twice in `engine.py` while this branch was open, so a line citation
here would be stale before it merged.)

## Gap 2: the step a score was reached at

The first fix was not enough, and the reason is worth recording because I had the wrong
gap in mind. My reading was that `metrics` already held both factors —
`secs_per_step_median` and `gsm8k_after` — so what was missing was the *product*.

Wrong. `gsm8k_before` / `gsm8k_after` answer "did this run cross a threshold". They cannot
answer **at which step**, and that step is `steps_to_score`. Moving the threshold to the
reading end does not help: whichever threshold a reader picks, the record does not contain
the moment it was crossed. (`tilerl-27` located this; my "the product is missing" framing
would have shipped a field that adds no information.)

So the GRPO loop now scores a fixed held-out subset every `--eval-every` steps and keeps
`{step, correct, total, score, secs}`. Four properties, each with a reason:

- **The subset is fixed across runs** — the first `--eval-curve-n` rows of `--eval-gsm8k`,
  not a sample. Two runs' curves are incomparable otherwise.
- **`secs` is summed, not `secs_per_step_median × step`.** The step length is not constant
  within a run; it changes once a run hits the rollout cap.
- **`secs` is training time and excludes the scoring.** `grpo_loop` stops its clock at
  `train.py:496`, before the yield, so the probe's cost is outside every point. That is the
  quantity the objective wants — production does not pay the probe — but it means the last
  point equals `secs_total`, not wall clock.
- **Scored at the yield, which is the only correct place.** `grpo_loop` calls
  `invalidate_weights()` at `train.py:492`, before yielding, so the eval sees the policy the
  step just produced with the decode graph already dropped.
- **Each point prices its own scoring** (`eval_secs`). The criterion for the subset size is
  "the eval costs under 5% of a step", and I was about to compute that bound from another
  run's recorded eval duration — which extrapolates across `n`, generation length and batch
  shape, and would have become a *default value*. **An estimated default is harder to
  overturn than no default, because it looks calibrated.** Recording the per-point cost makes
  the criterion a fact checkable after the first run, and one that tracks the model and `n`
  without maintenance. (`tilerl-27` stopped the estimate.)

Off by default (`--eval-every 0`), so no existing invocation changes shape.

## Gap 3: nothing could read it back

The threshold belongs at the reading end — the ledger records scores, and which one counts
as "the score" is a question about the objective, not about the record. But a record nothing
reads is not a measurement either, and after the first two gaps closed there was still no way
to get `time_to_score` out of a manifest.

`ledger.time_to_score(m, target)` and `tilerl ledger --time-to-score SCORE`. Three distinct
answers, because collapsing any two loses a fact a reader needs:

| case | answer |
|---|---|
| crossed | the point that crossed, plus the interval `(after_step, step]`, and `secs` there |
| instrumented, never crossed | `reached=False` with `best`, so the last point is never mistaken for the target |
| no curve | `None` — "not instrumented" is not "instrumented and fell short" |

**No interpolation, and this is the one judgement in the whole change.** The target almost
always falls *between* two scoring points, and interpolating gives a step nobody measured.
This figure is the project's headline metric; a fitted headline is precisely what the curve
was built to prevent. So the answer is the measured right end plus the interval — `step 30`,
`in (20, 30]` — and the reader can see how wide the uncertainty is instead of being handed a
false precision. (Judgement owed to `tilerl-27`, who named it before the reader was written.)

Verified against three mutants, each red: interpolating the midpoint, returning a miss for a
run with no curve, and returning the last point as if it had reached the target.

### The subset is a different quantity from the gate, and at n=20 it is noise-limited

`tilerl-0a` caught this on review: the curve scores a **subset**, so `score 0.55` on 20 rows
is not the `gsm8k_after` that P1's exit gate reads over `--eval-n` 500 rows. They named the
resolution — 20 rows is 5 pt per cell, and one cell is P1's whole target effect.

Computed rather than accepted, and it is worse than stated. The binomial SE at p=0.5:

| curve `n` | resolution | SE |
|---:|---:|---:|
| 20 | 5.00 pt | **11.18 pt** |
| 50 | 2.00 pt | 7.07 pt |
| 100 | 1.00 pt | 5.00 pt |
| 500 | 0.20 pt | 2.24 pt |

At n=20 the SE is **2.2x P1's +5 pt target**, so the *crossing step* is set by which rows
landed in the subset as much as by the policy. That is not a display problem — the returned
step is the headline number and it is sampling-limited.

So `n` and `se_pt` travel with every answer, and the reader prints `correct/total` plus a
note **only when `se_pt >= 5.0`** — P1's own target effect. Silent at 500 rows (2.24 pt),
loud at 20. A note on every line would be read as boilerplate and skipped, which is how a
real warning gets lost.

Three more mutants, all red: dropping the square root from the SE, silencing the note
unconditionally, and firing it unconditionally.

## The bug the pair-assertion caught, and the one a presence check would have missed

Both tests assert two things, and in both cases the second assertion caught a real defect
within minutes of being written.

**The engine test.** A key-set check alone passes over a hardcoded dict, so it also asserts
`blocks` tracks the context. First version compared `--max-new-tokens 4` against `200` and
**both arms reported 520 blocks** — `ctx` has a 1024 floor (`cli.py:568`) and 200 is under
it. 4 vs 4000 separates them (520 vs 2056).

**The curve test.** `secs` is asserted monotone and equal to `secs_total` at the last point,
not merely present. It had to be: my accumulator was named `elapsed`, and the timings loop
eight lines below is `for phase, elapsed in timings.items()`, which rebinds it every step.
Measured before the rename: **0.143 s at step 4 against 0.148 s at step 2** — a cumulative
figure going *down*. A presence check passes that. A monotone check does not.

Both verified from the other side: replacing `engine.config` with a literal dict turns the
first test red; freezing `secs` at 0, freezing `step` at a constant, and writing the curve
with the flag off each turn the second one red.

## Rules

- **Comparability should not change the id.** A field recorded so two runs can be compared
  goes beside the id's inputs, never inside them — otherwise every configuration change
  becomes a different run and the ledger loses its idempotence.
- **Record the config off the built object, not the constructor's arguments.** Clamps and
  reservations make them disagree, and the argument is the one a reader does not need.
- **"Did it reach X" and "when did it reach X" are different records.** A before/after pair
  cannot be made to answer the second by any choice of threshold at the reading end.
- **A cumulative quantity is asserted monotone, not present.** Presence is satisfied by any
  number, including one produced by a name collision.
- **Report the measured point and the interval, never the interpolated one.** A headline
  metric that is fitted between two samples is a number nobody observed, and the interval is
  the honest statement of how well it is known.
