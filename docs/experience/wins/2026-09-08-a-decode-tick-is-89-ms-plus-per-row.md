# A decode tick costs 8.9 ms plus 3.7–4.5 ms per active row — 2026-09-08

**Status:** measured on card 3, 20 GRPO rollout steps, group 8, cap 6144, real GSM8K
prompts. Every number here is for that length distribution — see the last two sections.

## Context

The GRPO rollout submits one prompt's `group` completions together and drains until the
last finishes (`train.py:435-443`), so a row that stops early leaves its slot idle. Two
questions followed: how much of a step is that idle, and what would recovering it be
worth.

The second question turns entirely on whether a decode tick's cost depends on how many
rows are active. If it does not, packing the idle slots is nearly free and the upside is
`1/(1-idle)`. If it does, refilling trades fewer ticks for more expensive ones.

## What a tick costs

Fitting `wall = b·max + c·sum` — `max` is the tick count (the longest row sets it) and
`sum` is the total row-ticks — over steps 1-19, step 0 excluded because it carries the
shape JITs:

```
wall = 8.05 ms × max + 4.451 ms × sum        R² = 0.943      (n=19)
wall = 8.87 ms × max + 3.719 ms × sum        R² = 0.9993     (n=8, the first fit)
```

- **`b` ≈ 8–9 ms** — a tick's cost independent of occupancy. This is the weight stream:
  24.44 GB read once per tick regardless of how many rows are served from it.
- **`c` ≈ 3.7–4.5 ms** — the marginal cost of each active row in a tick. KV traffic scales
  with active rows × their contexts; the weights do not.

An intercept was fitted first and came out **negative** (−2.76 s over 20 steps), which is
unphysical, so the two-term form above is the one to read. On the first eight steps the
intercept was +0.39 s and looked like a real per-step constant; it was absorbing curvature
the small sample could not see.

**This refutes "a tick costs the same regardless of M" as a claim about ticks.** That
claim was supported by a GEMM sweep showing per-call time rising only 10% from M=1 to
M=8 — but that sweep measures the weight stream alone, which is the `b` term. The `c`
term is invisible to it.

The consequence is counterintuitive and visible in the raw table: **the shortest step is
the most expensive per tick and the longest is the cheapest.** Step 5 (max 313) runs
23.45 ms/tick at 4.81 average active rows; step 6 (max 5022) runs 14.49 ms/tick at 1.42.
A long step is long because its tail runs one or two rows for thousands of ticks, and
those ticks are cheap.

## The coefficients drift, and the first entry claimed they did not

This entry originally said: *"The coefficients are stable as points are added: at n=8 they
are 8.84 and 3.700, at n=9 they are 8.83 and 3.705. A model that fits by absorbing noise
moves when the data grows; this one does not."* Eleven more steps:

| n | `b` (ms) | `c` (ms) | R² |
|---:|---:|---:|---:|
| 8 | 8.87 | 3.719 | 0.9993 |
| 11 | 8.79 | 3.754 | 0.9993 |
| 15 | 7.90 | 4.300 | 0.9907 |
| 19 | 8.05 | 4.451 | 0.943 |

**`c` moved 20% and R² fell from 0.9993 to 0.943.** The stability test was run over one
extra point, which is the range in which any fit looks stable. Two points do not test a
model's robustness; they test whether the ninth point resembles the eighth.

The refill gain that follows moves much less — 1.67x at n=8, 1.54x at n=19 — because it
depends on the ratio of the two coefficients and they drift together.

## The idle itself

Pooled over 20 steps, weighted by wall clock rather than averaged per step:

```
idle = 1 − Σ sum / Σ (max × group) = 0.745
```

Per-step it ranges 39.9% to 85.1%, and the per-step median is 71.3% — below the pooled
figure, because the expensive steps are the emptiest and a per-step mean gives a 7 s step
the same weight as a 155 s one.

**Two different ratios get confused here, and they differ by exactly `group`.**
`Σmax/Σsum` = 0.4908 is the one the gain formula takes; `Σsum/Σ(max·group)` = 0.2547 is
occupancy, and `1 − it` is idle. Both are dimensionless and both land in 0–1, so a
0.414 reported as one and read as the other travelled through three sessions and one
report to the principal before the factor of 8 was caught.

## What refilling the idle slots is worth: 1.54x, ceiling 1.89x

The two terms behave differently under refill, and that is the whole answer:

```
today:     wall = b·max        + c·sum
refilled:  wall = b·(sum/slots) + c·sum
```

**`c·sum` is unchanged.** The tokens still have to be generated; refilling rearranges them
across ticks and removes none of the KV work. Only `b·max` shrinks, to `b·sum/slots`.

```
gain = (b·max/sum + c) / (b/slots + c)
```

At `Σmax/Σsum` = 0.4908 over 20 steps:

| pooled gain (slots=8) | **1.54x** |
| ceiling (slots→∞, denominator falls to `c`) | **1.89x** |
| fraction of the ceiling that 8 slots already reach | **82%** |

**Three wrong answers were computed on the way.** `1/(1-idle)` = 3.93x assumes tick cost
is occupancy-independent, which `c` refutes — and `1/occupancy` is the same error in
different clothing, which is how a peer's ceiling of 2.42x arose. Dividing 3.93 by the
3.07x occupancy penalty gives 1.28x, also wrong: it applies the penalty to the whole wall
clock including `c·sum`, the part refill cannot touch. A third figure came from converting
a pooled *idle* back into a max/sum ratio — but per-step `max/sum` ranges 0.208 to 0.703,
and the mean of a ratio is not the ratio of the means.

**So `idle` overstates what is recoverable.** Its denominator is slot-ticks; the quantity
that matters has a denominator of cost. They agree only when a tick's cost is independent
of occupancy — and `c` is exactly the measure of how much it is not. In step 1, 79.6% idle
corresponds to 55% of the variable cost being recoverable.

Three limits on the coefficients:

1. **8 rows is 1.7x outside the observed range** (1.19 to 4.81 average active rows).
2. **The regressor is wrong for the counterfactual.** `sum` counts row-ticks, but KV
   traffic follows Σ(each row's context length). Within one of today's steps the rows have
   similar contexts, so the two are proportional and the fit cannot tell them apart.
   Refilling deliberately mixes a fresh short row with old long ones — exactly the
   configuration where the proportionality breaks.
3. **`c` is not purely KV.** Sampling and scheduling scale with active rows too.

The 1.89x ceiling depends only on `c` and does not use the extrapolated 8-row tick.

## Every number above is GSM8K's length distribution

The same measurement on MATH level 5 (a peer's, same cap, same group size) gives
occupancy 0.704 against this run's 0.2547 — **2.8x more homogeneous**. Through the gain
formula that is **1.10x** at 8 slots and 1.05x at 16, against 1.54x here. The lever is
worth doing on one dataset and not on the other, and nothing about the measurement says
which one you are looking at.

Two further scope conditions on the ratio itself:

- **It depends on `group`.** Occupancy has `max × group` in its denominator, so widening
  the group lowers it mechanically. Comparing two occupancies measured at different group
  sizes compares the group sizes.
- **Any run with rows at the cap reports a ratio that is not the distribution's.** 9 of
  20 steps here hit 6144; those pool to 78.6% idle against 67.0% for the other 11 — the
  clamp *raised* the figure, because the capped steps are the long ones and long steps are
  already idle. On the peer's data the same clamp *lowered* it, because there whole groups
  hit the cap together and it flattened within-group spread instead of between-step
  spread. **One mechanism, two levels, opposite signs — so a capped ratio must be
  discarded rather than corrected in a direction you assumed.**

## Rule

**A cost model needs one term per mechanism, and a small sample has room for one fewer.**
The 0.39 s intercept was not a fixed cost; it was curvature with nowhere to go, and it
went negative once the sample doubled.

**Testing a fit's stability against one more point tests nothing.** A model that absorbs
noise moves when the data grows — over eleven more steps, not over one.

**A measurement's scope has to travel with its number**, and scope is not only "which
kernel" or "which M". Here it was the dataset, the group size, and whether the cap was
reached. None of the three is visible in the number.

