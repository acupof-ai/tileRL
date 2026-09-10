# A decode tick costs 8.9 ms plus 3.7–4.5 ms per active row — 2026-09-08

**Status:** the tick coefficients stand. **Every length-distribution number below is VOID**
— the probe that produced them fed the model unrendered prompts. The numbers are kept,
marked, and explained rather than deleted, because a deleted wrong number gets recomputed
the same way by the next person.

## The prompts were never rendered

`probe_rollout_tail.py` fed `tok.encode(question)` — the bare document, with no
`<|im_start|>user`, no open assistant turn, and (thinking off) none of the empty
`<think></think>` the template closes in the prompt. `grpo_loop`'s prompts come from
`render_chat` (cli.py:611). A chat-tuned model handed a bare document continues the
document, so the rows ran long: mean 1083 tokens, 11 of 160 at the 6144 cap. An
independent measurement of the same dataset through the template read **mean 322, p90 532,
1.3% at a 1024 cap** — 19x on the mean, at 95.3% accuracy.

Void as a result: the pooled idle 74.5%, `Σmax/Σsum` 0.4908, the refill gain and its
ceiling, the train padding factor, and the observation that 12 of 18 steps had a row reach
5000 tokens — which is what prompted a degeneration investigation into a phenomenon the
probe had manufactured.

**What made it invisible was the part that was done right.** The sampler came from
`prompt.sampling` through `untruncated()`, exactly as `grpo_loop` builds it, and the probe
printed it on line two of every run. A correct, prominent, verified parameter set reads as
evidence that the rest of the input is correct too. The docstring argued at length for
real prompts over synthetic ones — it defended the data's *source* and never mentioned its
*format*.

Two things now guard it: the prompts go through `render_chat`, and every row is scored
with `answer_match`, with the pooled accuracy printed and a warning below 50%. An accuracy
column would have shown this in the first step.

**A second difference, which is not a defect.** `untruncated()` (train.py:429) drops top-p
and top-k because `rl_step` differentiates the full softmax, so this probe samples all
248320 tokens where an eval samples top_p 0.8 / top_k 20. Its lengths should therefore
exceed an eval's on the same prompts by an unknown factor. The 322 is not a target to
reproduce; it is a floor.

## What survives

- **`b` and `c`** — fitted against each step's `max` and `sum` whatever produced those
  lengths. The tick sweep measures them directly and does not use prompts at all.
- **Throughput 1.325x for group 8 → 16** — both arms drew from the same wrong
  distribution, so the ratio holds, but it is now a ratio measured in a regime the
  training path never enters.

The sections below are preserved as written. Read the tick-cost and R² sections; treat
every occupancy, idle, and refill figure as void.

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

## R² = 0.943 is also an artifact, and the honest number is 0.759

Fitting `wall` weights each step by its `max`, because a step with 6144 ticks contributes
a hundred times the sum-of-squares of a step with 313. The long steps are exactly the ones
with the fewest active rows — 1.19 to 2.5 — so the fit is optimised on the region where
`c` barely matters, and reports the fit quality of that region.

Regressing the quantity the model actually claims, `ms/tick` against average active rows,
one point per step, unweighted:

```
ms/tick = 9.79 + 3.472 × avg_rows        R² = 0.759
```

> **Provenance (2026-09-10 cleanup):** the one-off `scripts/probe_tick_cost.py` was deleted here; rerun it by hand with `scripts/pod_run.sh --wait ticks <card> -- python3 scripts/probe_tick_cost.py`. No code replaces the instrument: the ms/tick vs active-rows fits are card measurements; this entry is the record, rebuild on a card from the command.

**A quarter of the variance is unexplained.** Adding a `rows × context depth` interaction
takes it to 0.794 and says `c` rises 22% from depth 512 to depth 3072 — real, and far too
small to reconcile the disagreement in the next section.

So the two-term model is a usable first-order description and not much more. **Everything
below that extrapolates it to 8 or 16 rows is extrapolating a model that explains 76% of
its own data over a range it never observed** (1.19 to 4.81 rows).

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

**A probe that claims to reproduce a shipped path has to enumerate what that path does
before the step it reproduces.** This one took the sampler from `prompt.sampling` and the
prompt from nowhere; `render_chat` sits one layer outside the `tok.encode` it copied. List
the shipped call chain and tick off each stage, rather than checking that the one stage
you thought about matches.

**A distribution measured with no correctness column cannot tell a tail from garbage.**
Length is cheap to record and means nothing on its own. Whatever a probe measures about
generated text, it should also score the text — here one `answer_match` per row would have
read near zero on the first step and saved the day's four downstream investigations.

