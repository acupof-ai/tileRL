# Speculation loses on sm70: the draft step is outside the graph — 2026-08-31

> Status: **Superseded.** The draft loop does run outside the graph, but that is
> not the dominant cost. See
> [2026-08-31-m8-gemv-occupancy-not-reuse.md](2026-08-31-m8-gemv-occupancy-not-reuse.md).

## Context

Dense B=1 decode is 25.8 tok/s on the V100 and the weight-bandwidth roofline is
56.1 tok/s (16.04 GB streamed / 900 GB/s = 17.8 ms/token; this entry originally
cited a remembered 14 GB / 64 tok/s —
`errors/2026-09-02-roofline-is-the-streamed-subset.md`), so 60 tok/s is
unreachable without speculation. The checkpoint ships an MTP head, so the head
itself was free.

Measured MTP quality (`scripts/check_mtp_draft.py`): **62% top-1 agreement**
with the trunk, median trunk-rank of the draft's pick = 0, 84% inside the
trunk's top-5. Geometric extrapolation said depth 6 → 62.5 tok/s.

Measured reality: **3.1 tok/s at depth 6** — 8× SLOWER than no speculation.

## Root Cause

Acceptance was never the problem. `/health` showed 97-99% accept and **5.33
tokens committed per forward** — the policy works exactly as designed. The cost
is the draft.

`scripts/prof_draft_step.py` timed the two forwards separately and broke the
assumption the whole plan rested on:

```
trunk forward (M=1):  103.58 ms      <- EAGER. The captured tick is 39 ms.
draft step    (M=1):  120.91 ms      <- 1.17x the trunk, for a 1-layer head
bandwidth floor for a 456 M-param head: 0.25 ms fp4, 1.01 ms bf16
```

A 1-layer, 456 M-param head costs **as much as the whole 64-layer trunk**, and
480× its own bandwidth floor. Both are launch-bound at M=1: each projection is
one row, so the time is kernel launch, not arithmetic.

The trunk hides this behind graph capture (eager 103.58 → captured 39 ms, worth
2.66×). **The draft step runs outside the captured region** — `_draft_step`
walks the head autoregressively, one step per token — so it pays the eager
price every step:

| depth | tick (39 + d×120.91) | tokens | tok/s |
|---:|---:|---:|---:|
| 1 | 159.9 | 1.62 | 10.1 |
| 2 | 280.8 | 2.00 | 7.1 |
| 4 | 522.6 | 2.39 | 4.6 |
| 6 | 764.5 | 2.54 | 3.3 |

Predicted 3.3 at depth 6, measured 3.1. The model is right.

## Two wrong turns, recorded because the reasoning looked sound

1. **"fp8 quantization has no sm70 kernel"** — true (`linear_fp8` is
   sm90-only, so `_quantize_draft` sent every projection to the torch
   fallback), and fixing it took 0.7 → 6.0 tok/s. But that is 8.6× on the
   wrong axis: still 4× worse than no speculation.
2. **"then quantize to fp4, sm70's fused format"** — made it *worse* (3.1 vs
   6.0). At M=1 the format is irrelevant because nothing is bandwidth-bound;
   the fp4 GEMV just adds dequant work per launch.

Both were inferred from end-to-end throughput. Neither survived a direct
measurement of the draft step. The `has_kernel` probe from (1) is kept — a
format should only be produced where a kernel consumes it — but it was not the
bottleneck.

## Fix

Capture the draft step. At its bandwidth floor the arithmetic inverts:

| depth | tick (39 + d×0.25) | tokens | tok/s |
|---:|---:|---:|---:|
| 2 | 39.50 | 2.00 | 50.7 |
| 4 | 40.00 | 2.39 | 59.8 |
| 6 | 40.50 | 2.54 | **62.7** |

Not yet implemented. `_DecodeGraph` captures `model.forward` at a fixed `(B, W)`;
the draft loop needs the same treatment, which is harder because it is
sequential and reads its own previous hidden.

## Also found

`step_states` is `[num_slots, L, spec_steps, heads, K, V]` — sized by SLOT
COUNT, not by live requests. At 16 slots and depth 6 that is
`16 × 7 × 144 MiB = 15.75 GiB`, which OOM'd a 32 GB card outright. Dropped to
4 slots (3.94 GiB) for the B=1 measurements.

Tree verification (the original plan) is blocked separately:
`kernels_gdn.py:500-520` evolves `state_local` across the `t` loop, so node t's
state builds on t−1, **not on its parent**. A tree needs per-node forking from
the parent state. Deferred — a linear chain's ceiling is
`1 + p/(1−p) = 2.63` tokens = 67.5 tok/s at p=0.62, which clears 60 without a
tree.

## Rule

A draft model is only cheap if it runs where the trunk runs. Speculation moves
work from one big launch-bound forward into many small ones, so on any arch
whose decode is launch-bound rather than bandwidth-bound, the draft must be
captured/fused before acceptance rate matters at all. Measure the draft step in
isolation before tuning anything about it — end-to-end throughput cannot
distinguish "bad acceptance" from "expensive draft", and two plausible
format-level diagnoses both survived until the step itself was timed.
