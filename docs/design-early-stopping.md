# Early stopping for the GRPO curve: four decisions — 2026-09-08

**Asked by:** tilerl-27, after the four-point curve landed (base 87.4 / step 25 93.2 /
step 50 **93.4** / step 75 82.4 / step 100 91.2). Stopping at the peak is worth **1.99x wall
clock and +2.2 points** against running to 100.

**Read Q4 first. It changes what the other three are worth.**

---

## Q4 — The weights at step 50 do not exist

**`cli.py:961-967` is the only writer of adapter tensors, and it runs once, after the loop.**
There is no periodic checkpoint, and nothing in `src/` mentions `patience`, `best_score` or
`early_stop` — grep returns nothing.

So today "stop at step 50" and "run to 100 and roll back" are not two implementations of one
idea. **The second is impossible**, because the step-50 weights were overwritten in place:
`AdamW.step_one` ends in `p.copy_()`, which is exactly the property that lets the engine keep
its captured graphs. Every intermediate policy is destroyed by the next step.

**This inverts the ordering of the work.** Early stopping cannot be a consumer of the curve
until a snapshot exists; without one, the only stop that saves anything is a stop *at the
moment of the decision*, which means the decision must be made online, from the point just
scored, with no ability to go back.

**The snapshot is cheap and should land first, independent of any stopping policy.** Live
LoRA is 124.84 M params — 238 MiB at bf16 — and a device-to-host copy of that is tens of
milliseconds against a 20.67 s step. Keeping **one** host-side copy of the best-scoring point
costs one buffer and no disk. Writing every point to disk costs 4 × 341 MB per run and buys
the ability to change the selection rule after the fact.

**My recommendation: keep the best point in host memory, write it to disk only at the end,
as `adapter-best.safetensors` beside the existing `adapter.safetensors`.** Two files, and the
manifest says which step each came from. That makes "roll back" real without making anyone
choose a stopping rule yet, and a run that ends normally still ships the final adapter it
ships today.

---

## Q1 — The signal has to be the eval curve, and it costs more than what it gates

Training reward is out, and this run is the proof: **through the 11-point collapse at step
75 the reward fell about 5% and held.** A signal that does not move when the thing you care
about drops 11 points is not a signal.

That leaves the curve, and its price is the design problem:

| `--eval-every` | points | eval wall | train wall | eval share |
|---:|---:|---:|---:|---:|
| 25 | 4 | 41 min | 34 min | **54%** |
| 10 | 10 | 102 min | 34 min | 75% |
| 5 | 20 | 205 min | 34 min | 86% |

**One eval point (614 s) costs more than the 25 training steps it evaluates (537 s).** At the
current cadence the instrument is already the majority of the run, and "stop earlier" competes
directly against "measure often enough to know when to stop."

Two levers, and they are not symmetric:

- **`--eval-curve-n` is 20 rows** (`cli.py:1481`), which is where the 614 s comes from and
  also where the noise comes from — #325 computes the per-point SE from the observed `p`
  rather than 0.5. A stopping rule reading 20 rows is reading a quantity with a wide band;
  raising `n` sharpens the decision and raises the cost linearly.
- **`at_cap` and `mean_len` now travel with every point** (#323). A drop with `at_cap` rising
  is truncation, a drop with `at_cap` flat is the policy. **These are free** — already
  computed — and they change what a dip means without costing a second.

**Decision: the signal is the curve's score, qualified by `at_cap`. Do not add a cheaper
proxy.** A cheaper signal that disagrees with the curve is worse than no signal, and the one
cheap signal available (training reward) is already known to have missed the event this
feature exists for.

---

## Q2 — Patience is not determinable from this run, and I will not invent a number

The curve is 93.4 → 82.4 → 91.2. Both candidate rules are wrong on it:

- **patience = 1** stops at step 75, in the hole. It would have shipped 82.4% — worse than
  base. It also happens to keep the step-50 weights, so with a snapshot it ships 93.4%: the
  *stop* is wrong and the *selection* is right.
- **patience = 2** never triggers; the run goes to 100 and saves nothing.

**One collapse-and-partial-recovery is one observation.** It cannot separate "dips are common
and recover" from "this run hit one bad update." Fitting a patience to it would be exactly
the `f = 0.3` failure — a parameter with a real derivation from a curve that happens to be
the only one we have.

**Decision: ship no patience. Ship best-point selection instead**, which is well-defined on
this data (the peak is step 50, unambiguously) and does not require predicting the future.
Patience becomes answerable after N runs show whether a dip typically recovers; that is a
question for a population of curves, not for this curve.

**What this costs, stated plainly:** best-point selection alone saves **zero wall clock**. It
buys the +2.2 points, not the 1.99x. The 1.99x needs a stopping rule, and a stopping rule
needs data we do not have.

---

## Q3 — Where it hangs: a consumer of existing fields, not a new mechanism

Everything needed is already in the tree or in flight:

- `score_curve` (`cli.py:840`) computes the point and appends it to `curve`.
- `_time_to_score` (`ledger.py`) already walks the points; #325 adds **`held`** and
  **`dipped_at`**, which is precisely "did every later point stay above the target" — the
  post-hoc form of the same question a stopping rule asks online.
- The loop at `cli.py:898` already calls `score_curve(i + 1)` at the right moment.

**Minimal change: `score_curve` snapshots `trainable` when the new point is the best so far.**
That is a few lines inside a function that already exists, running at a cadence that already
exists. No new flag, no new file format, no new loop.

**Do not build a stopping mechanism now.** With the snapshot in place, a stopping rule is a
one-line predicate added later against a field that will already be populated. Building the
policy first, on one curve, is the expensive ordering.

---

## Summary

| question | answer |
|---|---|
| Q4 checkpoint | **Does not exist.** `cli.py:961` writes once, post-loop; `p.copy_()` destroys every intermediate. **Fix this first.** |
| Q1 signal | The curve's score, qualified by `at_cap`. Training reward is disqualified — it held through an 11-pt collapse. Cost: one point (614 s) exceeds the 25 steps it measures (537 s). |
| Q2 patience | **Undeterminable from one curve.** patience=1 stops in the hole, patience=2 never fires. Ship best-point selection, which needs no forecast. |
| Q3 attachment | `score_curve`, as a snapshot on a new best. Consumer of existing fields, not a mechanism. |

**The honest bottom line:** the snapshot buys the +2.2 points now and is small. The 1.99x
needs a stopping rule, and this run cannot tell us what that rule is — it can only tell us
that reward is not the signal and that the peak is recoverable if someone saved it.
