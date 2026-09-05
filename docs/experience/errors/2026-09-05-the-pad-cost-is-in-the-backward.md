# The pad cost is in the backward, and my arithmetic was wrong in both directions

**Date:** 2026-09-05
**Source:** H20 sm90 card 6, Qwen3.8-27B-NVFP4, one `grpo` step, group=8 `gen=64`.
Every `_pad2d` call instrumented and bucketed by call site and by `M`.

## Context

The [09-03 profile](2026-09-03-grpo-step-is-python-dispatch-bound.md) put
`torch.nn.functional.pad` at **14.77% self**, top of the table, and read it as
"the pad is recomputed on every linear in every layer of every forward". I argued
that reading assumed the padded tensor is the *weight*, computed which tensor
actually needs padding at 27B dims, and concluded the real cost is the
`M=8 → Mp=16` activation row pad in the rollout.

Then I measured it. **Both readings are wrong, and mine is wrong in two directions
at once.**

## What the histogram says

One step: **175,424 real pads copying 40.1 G elements**, plus 192,344 no-ops —
52% of all `_pad2d` calls already return their argument untouched, so the early
return at `backend.py:65` is doing real work.

| | elements | calls |
|---|---:|---:|
| forward (`model.py:*`, `backend.py:*`) | 14.4 G | 121,896 |
| **backward (`autograd.py:*`)** | **25.7 G** | 53,528 |

By `M`:

| M | elements | what it is |
|---:|---:|---|
| **74** | **23.0 G** | the backward's padded batch — **57% of everything** |
| 17408 | 5.7 G | weight rows |
| 8 | 4.8 G | the rollout row pad I predicted was the lever — **12%** |
| 5120 | 1.6 G | weight rows |

`M=74` appears at **four sites, all of them in `autograd.py`** — `_linear`,
`wrapper`, `master_linear`, `handler` — and at **zero** forward sites.

## The two things I got wrong

**1. I said the fix is to allocate rollout activations at `Mp` rows up front.**
That addresses the `M=8` bucket: 4.8 G of 40.1 G, 12%. The volume lives in the
backward, which I never modelled. I assumed the backward ran at M=256 (where every
row pad is a no-op, since 256 is already a multiple of the tile) and it runs at
M=74 — the padded batch of 8 rollouts at this generation length, not the sequence
length I pictured.

**2. I said every real pad is an activation pad.** Activation-shaped M (8, 16, 48,
74) is 29.9 G = 74.5%; weight-shaped M (≥512) is 10.2 G = 25.5%. So weight pads
are a quarter of the volume — not zero, as my first analytic pass said, and not
the ~38% my corrected pass said either.

The analytic pass was arithmetic over config dims predicting which calls take the
early return. It was directionally useful — it correctly killed `_const_arg`, since
caching a *derived weight tensor* cannot touch a backward activation pad — and it
was quantitatively wrong about everything it estimated.

## No fix proposed

40.1 G of elements copied is a **volume**, and 14.77% is a **time**. Nothing here
shows the time distributes the way the volume does, and a lever chosen from the
volume alone would be the same mistake in a new place. The next step is #101's
phase-attributed seconds (rollout vs backward) on the MATH run, read *against*
this histogram. `rowpad` closes until then.

## Rule

**Arithmetic over config dims predicts which branch runs, not how much it costs.**
Every prediction I made from the shapes was checkable and each one was off: weight
pads "all no-op" (they are 25%), the backward "all no-op at M=256" (it is 57% at
M=74), the rollout pad "the lever" (12%). The predictions were cheap and worth
making — they killed a wrong fix before it was written — but a bucketed count from
one real step cost one run and corrected all three.

Corollary, from the same run: **a profile's call-site attribution names the
function, not the tensor.** `_base_linear -> linear_fp4 2.70%` says where `pad` was
called from; it says nothing about whether the thing being padded was a weight, an
activation, or a gradient, and that distinction is what decides the fix.
