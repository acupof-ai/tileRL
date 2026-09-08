# A wrong reason kept a 2.6x decode gap unexamined — 27B, 2026-09-08

**Status:** open — the gap is measured and reproducible; 93% of it is unattributed.

## Context

The GRPO step is 73.8% rollout
([the step is 74% rollout](../wins/2026-09-07-the-step-is-74-percent-rollout.md)), so the
rollout's per-tick cost sets the RL wall clock. Two full-27B B=8 decode measurements in this
tree disagree by 2.64x, and the slower one has the *shorter* context.

## The measurement

Both numbers are wall-clock milliseconds per decode forward. `ab_draft_depth.py:303` returns
`wall / (decode_forwards delta)` and exits non-zero if any mixed tick lands inside the window
(`:291`); the rollout figure is `rollout_secs / 1024`, and
[a16ff9c](../wins/2026-09-06-one-grpo-step-is-54-percent-backward.md) independently attributed
1023 decode ticks to 56.484 s of the same rollout. Same numerator, same divisor, no mixed ticks
either side — the gap is not a denominator artifact.

| arm | sha | card | ms/tick | graph |
|---|---|---|---:|---|
| training rollout | a16ff9c | 6 | 55.21 | on |
| training rollout | 73433bb | 0 | **61.68** | on |
| serving, W=1 no draft | d1f2bb8 | 6 | **23.34** | on |

The two training numbers agree to 1.12x across different shas *and* different cards, and both
are 2.37–2.64x the serving arm. So neither the card nor the 09-06 arm's partly-drained batch
(`tok/fwd` 6.90, not 8) can produce it: a run-to-run artifact does not reproduce twice.

## The reason on record was false, which is why nobody looked

`one-grpo-step-is-54-percent-backward.md` explained the 55.21 ms/tick as "the training path
(`_training_kv` builds a dense pool, no draft plane, no `spec_depth`)". Both halves are wrong.
`_training_kv` is called at `train.py:160`, inside the forward/backward — the rollout is
`engine.submit` plus `_drain` (`train.py:442-445`), the ordinary paged engine, the same path
serving uses. And the serving arm was itself W=1 no-draft, so the draft plane separates nothing.

A 2.4x sat unexamined for two days behind a sentence that read like an explanation. That is worse
than an unexplained number: an unexplained number invites a probe.

## What is priced, and it is 7%

| candidate | worth | share of the 38.34 ms gap |
|---|---|---|
| LoRA adapters (`model.py:256`, rank 16) | ≤1.54 ms | 4.0% |
| `fuse_projections` — serving `True`, training `False` | +4.8% measured | 2.9% |
| **accounted** | **2.66 ms** | **6.9%** |

LoRA is refused on two independent bounds rather than measured: 0.537 GFLOP/tick of extra work is
0.054 ms even at a terrible 10 TFLOP/s, and its 768 extra launches sit inside a *captured* graph
where replay is 1–2 µs. The fusion difference is real but deliberate — serving-only, because
training keeps the unfused bf16 masters
([projection fusion](../wins/2026-08-25-projection-fusion-decode.md)) — and that entry measures
it at 1.821 → 1.734 ms.

Ruled out **by direction, not magnitude**: the serving arm ran ctx 2048 with 128 new tokens
(KV 2048→2176) where the rollout ran prompt 256 with gen 1024 (KV 256→1280). The rollout's KV is
smaller throughout, so attention cost cannot produce a gap in that direction at any size.

Ruled out by source: the decode graph is on in the measured step (`step_phase_split.py:65`,
`:88`), and the graph is worth 2.16x when flipped off→on
([recapture after update](../wins/2026-09-05-recapture-after-update.md)). An eager-dispatch
penalty would have shown up as that 2.16x, and it is already banked.

## Root cause

Unknown. 93% of a gap worth **45.9% of a GRPO step** if it closed — the largest unexplained
quantity in the project.

## Fix

Not a fix yet, an arm. One process, one card, one script, two engines: the rollout's
`build_engine(...)` against the serving arm's, same context schedule, both no-draft, both graph
on. That isolates the engine-configuration bundle — `num_blocks`, `max_total_tokens`,
`NoPrefixStore`, `num_slots`, sampling — which is where the gap has to live now.

A reproduction there is a **localization, not a mechanism**: it says the cause is inside those
five, not which one, so plan a second bisect round and report it as localized. (Framing owed to
`tilerl-27`.)

## Rule

A caveat that names a mechanism is a claim, and it is load-bearing in a way a number is not: a
wrong number gets re-measured, a wrong reason closes the question. Before writing "X is not
comparable to Y because Z", read Z — `_training_kv`'s own call site would have taken one grep.

Two measurements of the same quantity that disagree by more than 2x are a defect in one of them
or a finding in both. Neither is a footnote.
