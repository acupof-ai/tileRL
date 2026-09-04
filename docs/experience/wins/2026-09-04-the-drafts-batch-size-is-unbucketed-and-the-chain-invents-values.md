# The draft's batch size is unbucketed and the chain loop invents values the batch never had

**Date:** 2026-09-04
**Arch:** target-independent (measured on the CPU twin — this is dispatch arithmetic, not a
kernel property)
**Task:** #74 follow-up
**Instrument:** `scripts/probe_draft_batch.py`
**Verdict:** The draft hands its kernels **6 distinct batch sizes** at `max_batch=8` (7
across every arm), of which **2 come only from the chain loop**. Bucketing is free at every
endorsed width (previous entry), so this is a real saving — but it needs the draft's own rung
behaviour measured on hardware before it ships.

## Why this was open

[The B-axis entry](2026-09-04-b-is-a-shape-axis-and-decode-already-buckets-it.md) found that
`engine.py:817` already buckets B on the decode graph path, and that `spec.py` contains the
string "graph" **zero** times — the draft always runs eager. Two sites take B raw:

- `spec.py:414` `n = len(plan)` — the pooled prefill
- `spec.py:474` `len(live)` — the chain loop, which *shrinks* as rows hit block boundaries

## Measured

Staggered arrivals so the batch forms and drains, depth 3, tiny config:

| max_batch | reqs | prefill n | chain n | distinct |
|---:|---:|---|---|---:|
| 1 | 3 | {1: 21} | {1: 40} | 1 |
| 2 | 4 | {2: 14} | {1: 2, 2: 26} | 2 |
| 4 | 6 | {2: 9, 4: 6} | {2: 18, **3: 2**, 4: 10} | 3 |
| 8 | 10 | {2: 6, 4: 2, 6: 3, 8: 4} | {2: 12, **3: 2**, 4: 2, 6: 7, **7: 1**, 8: 6} | **6** |

Across all arms: **n ∈ {1, 2, 3, 4, 6, 7, 8}** — 7 values, of which the `max_batch=8` arm
alone contributes 6 (it never forms a single-row batch).

**n=3 and n=7 never appear at the prefill site.** They exist only because the chain loop
drops rows mid-chain — `spec.py:476` breaks or narrows when a row has no block room left,
and the `ponytail:` comment there says so: "a row at a block boundary drafts shorter". So
the draft compiles shapes the batch itself never formed, and no `max_batch` setting predicts
them.

At `max_batch=8` that is 6 specializations where bucketing to powers of two gives 3
(2, 4, 8) — and the previous entry established the padding is free at every width the
sm70 ladder endorses.

## What is NOT concluded

The compile count is not converted to a time saving here. The draft's rung behaviour was
never measured on hardware: every rung number in this line
(`wins/2026-09-04-rung-cost-not-useful-rows.md`) is the trunk's, and the draft is a 1-layer
head with its own occupancy. Borrowing the trunk's rung table for it would repeat exactly
the error the previous entry withdrew — using a constant without the model it came from.
The card is unavailable (the user's 27B server holds it), so this stops at the axis count.

## Two instrument corrections

**1. The site delimiter was a shape guess and it was wrong.** I split prefill from chain
with `ids.shape[1] > 1 or not seen`, reasoning the prefill is wider. At depth 1 the prefill
is also 1 wide, and `not seen` is true only for the very first call ever, so nearly every
prefill was counted as a chain step: **2 prefills against 22 chain calls**, when every step
makes exactly one prefill. Wrapping `step` to delimit the sites fixed it to 8 and 16.

**2. My replacement invariant was too strict, and the assert earned its keep.** Having fixed
the delimiter I asserted `chain == prefills × (depth−1)`. It failed at 21 prefills / 40 chain
calls — not because the delimiter was broken again, but because the early break is real. The
correct invariant is `0 < chain ≤ prefills × (depth−1)`, and the failure is what sent me to
read `spec.py:474-476` instead of trusting the arithmetic I expected.

## Rule

A count taken at one call site is not the axis. The chain loop's `len(live)` produced two
values the batch never had (n=3, n=7), so counting only where the batch is *chosen* would
have found 5 specializations across the arms and missed 2 — and those 2 depend on
block-boundary alignment, which no configuration knob exposes.

And an invariant that fails is worth more than one that passes: both of this probe's numbers
were wrong until an assert rejected them, first on a bad delimiter and then on a real
early-exit path I had not read.
