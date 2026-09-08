# A decode tick costs 8.84 ms plus 3.70 ms per active row — 2026-09-08

**Status:** measured on card 3, 8 GRPO rollout steps, group 8, cap 6144, real GSM8K
prompts. The refill upside that follows is **not** established — see the last section.

## Context

The GRPO rollout submits one prompt's `group` completions together and drains until the
last finishes (`train.py:435-443`), so a row that stops early leaves its slot idle. Two
questions followed: how much of a step is that idle, and what would recovering it be
worth.

The second question turns entirely on whether a decode tick's cost depends on how many
rows are active. If it does not, packing the idle slots is nearly free and the upside is
`1/(1-idle)`. If it does, refilling trades fewer ticks for more expensive ones.

## What a tick costs

Fitting `wall = a + b·max + c·sum` over 8 steps — `max` is the tick count (the longest
row sets it) and `sum` is the total row-ticks:

```
wall = 0.39 s + 8.84 ms × max + 3.700 ms × sum        R² = 0.9993
```

- **`b` = 8.84 ms** — a tick's cost independent of occupancy. This is the weight stream:
  24.44 GB read once per tick regardless of how many rows are served from it.
- **`c` = 3.70 ms** — the marginal cost of each active row in a tick. KV traffic scales
  with active rows × their contexts; the weights do not.
- **`a` = 0.39 s** — a per-step fixed cost, small.

The coefficients are stable as points are added: at n=8 they are 8.84 and 3.700 ms with
R² 0.9993, at n=9 they are 8.83 and 3.705 with R² 0.9993. A model that fits by absorbing
noise moves when the data grows; this one does not.

A two-point fit on the same data had put the fixed term at 2.99 s, because two points
cannot separate a per-step constant from a per-row slope; the row term was absorbed into
the intercept. Three parameters over eight points separate them, and `max` and `sum` are
not collinear here (step 2 is 663/2163, step 6 is 5022/7143).

**This refutes "a tick costs the same regardless of M" as a claim about ticks.** That
claim was supported by a GEMM sweep showing per-call time rising only 10% from M=1 to
M=8 — but that sweep measures the weight stream alone, which is the `b` term. The `c`
term is invisible to it.

The consequence is counterintuitive and visible in the raw table: **the shortest step is
the most expensive per tick and the longest is the cheapest.** Step 5 (max 313) runs
23.45 ms/tick at 4.81 average active rows; step 6 (max 5022) runs 14.49 ms/tick at 1.42.
A long step is long because its tail runs one or two rows for thousands of ticks, and
those ticks are cheap.

## The idle itself

Pooled over 8 steps, weighted by wall clock rather than averaged per step:

```
idle = 1 − Σ sum / Σ (max × group) = 0.754
```

Per-step it ranges 39.9% to 82.2%, and the simple mean is 0.672 — **8 points below the
pooled figure**, because the expensive steps are the emptiest and a per-step mean gives a
13 s step the same weight as a 155 s one.

**The cap accounts for part of it but not most.** Three of eight steps had a row hit the
6144 cap; those pool to 78.2%. The five that did not still pool to **69.7%**. The same
step-0 rows at cap 2048 would have shown at most 63.7% idle, so the figure moves ~16
points with the cap and must never be quoted without it.

## What refilling the idle slots is worth: 1.53x, ceiling 1.99x

The two terms behave differently under refill, and that is the whole answer:

```
today:     wall = a + b·max        + c·sum
refilled:  wall = a + b·(sum/slots) + c·sum
```

**`c·sum` is unchanged.** The tokens still have to be generated; refilling rearranges them
across ticks and removes none of the KV work. Only `b·max` shrinks, to `b·sum/slots`.

```
gain = (b·max/sum + c) / (b/slots + c)
```

Per step over ten steps, pooled on Σmax/Σsum = 0.414:

| pooled gain (slots=8) | **1.53x** |
| ceiling (slots→∞, denominator falls to `c`) | **1.99x** |

Per-step it ranges 1.15x to 2.06x, tracking each step's `max/sum`.

**Two wrong answers were computed on the way, and both are instructive.** `1/(1-idle)` =
4.07x assumes tick cost is occupancy-independent, which `c` refutes. Dividing that by the
3.07x occupancy penalty gives 1.33x, and is also wrong: it applies the penalty to the
whole wall clock including `c·sum`, the part refill cannot touch. A third figure, 1.70x,
came from converting a pooled *idle* back into a max/sum ratio — but per-step `max/sum`
ranges 0.208 to 0.703, and the mean of a ratio is not the ratio of the means, which is the
same error this entry documents elsewhere.

**So `idle` overstates what is recoverable.** Its denominator is slot-ticks; the quantity
that matters has a denominator of cost. They agree only when a tick's cost is independent
of occupancy — and `c` is exactly the measure of how much it is not. In step 1, 79.6% idle
corresponds to 55% of the variable cost being recoverable.

The three limits below still apply to the coefficients, though note the 1.99x ceiling
depends only on `c` and does not use the extrapolated 8-row tick:

1. **8 rows is 1.7x outside the observed range** (1.42 to 4.81 average active rows).
2. **The regressor is wrong for the counterfactual.** `sum` counts row-ticks, but KV
   traffic follows Σ(each row's context length). Within one of today's steps the rows have
   similar contexts, so the two are proportional and the fit cannot tell them apart.
   Refilling deliberately mixes a fresh short row with old long ones — exactly the
   configuration where the proportionality breaks.
3. **`c` is not purely KV.** Sampling and scheduling scale with active rows too.

Measuring an 8-row tick directly beats extrapolating, and it does not need this harness: a
synthetic rollout with all rows the same length holds occupancy constant by construction.

## Rule

**A cost model needs one term per mechanism, and a two-point fit has room for one.** The
2.99 s intercept was not a fixed cost; it was a slope with nowhere to go. Before reading a
coefficient, count the mechanisms you believe are present and check the fit has a
parameter for each.

**A measurement's scope has to travel with its number.** The GEMM sweep was correct and
was correctly quoted, and the conclusion drawn from it was still wrong, because "per-call
time barely moves with M" silently became "a tick costs the same at any M". The sweep
covered weights; the tick also reads KV.
