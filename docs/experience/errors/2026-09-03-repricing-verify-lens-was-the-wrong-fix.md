# Repricing verify_lens was the wrong fix — the cost's shape is wrong, not its scale, V100 sm70, 2026-09-03

> Status: **#38 REJECTED as diagnosed, and no constants changed.** Last tick I found
> `verify_lens` pricing sm70 with H20 constants (211.0 / 0.53 ms against a measured
> 15.9 / 12.5) and filed the fix as "reprice two constants, A/B it". Reading the code
> that consumes the trim shows the reprice would have **bought nothing at the shipped
> batch size and actively cut the widths that earn 1.157-1.228×**. The defect is real;
> my diagnosis of it was not.

## Context

[`wins/2026-09-03-verify-tick-cost-is-a-line-in-width.md`](../wins/2026-09-03-verify-tick-cost-is-a-line-in-width.md)
measured a verify tick at `0.670 + 0.5265·W` dense ticks. `spec.py` prices the draft trim
with the *same affine form*, `BIAS_MS + ROW_MS·r`, so the numbers looked directly
substitutable — and the substitution changed the trim's decision at every acceptance
level, which read as proof the mispricing mattered.

It does matter. Just not the way I said, and the fix I filed was wrong in three ways
that only reading `_run_forward` reveals.

## What the consumer actually does

`engine.py:684` — the width the tick pays for is the **widest chain**, not the sum:

```python
w = max(map(len, chains))
for c in chains:
    c.extend([c[-1]] * (w - len(c)))   # every chain padded to w
```

and `engine.py:679` — if the trim keeps nothing, the tick stops being a verify tick:

```python
if chains is not None and max(map(len, chains)) == 1:
    chains = None  # the policy kept nothing: a plain decode tick
```

Then `B·W` rows go through the sm70 ladder, which rounds **up**:

| W | B·W rows at B=4 | rows launched | cost (dense ticks) |
|---:|---:|---:|---:|
| 1 | 4 | 4 | 1.20 (+ draft forward skipped) |
| 2 | 8 | 8 | 1.72 |
| **3** | **12** | **32** | **2.78 — same as W=4** |
| 4 | 16 | 32 | 2.78 |

So three defects, and the reprice addresses none of them:

1. **The trim optimizes the wrong quantity.** Its denominator is `bias + row·(r+i)`,
   total rows admitted. The cost depends on `max(keep)+1`. Trimming a short chain while
   one long chain survives changes the objective and not the bill.
2. **The cost is a staircase, not a line.** Repriced, `verify_lens` moves a
   high-acceptance batch from keep=3 to keep=2 — W=4 to W=3 — which launches the **same
   32 rows**. A "fix" whose entire visible effect is a no-op at the shipped `max_batch=4`.
3. **Where it does bite, it bites the wrong way.** The measured price puts the
   keep≤1 boundary at acceptance **p≈0.92**, and the recorded acceptance is **84.4%**
   (`engine.py:296`) — just below it. So the repriced trim would force W≤2 at exactly the
   contexts where depth 3 measured **1.157× (512)** and **1.228× (1024)**, against depth
   1's 1.133× and 1.112×. It would have cost throughput to fix a mispricing.

## The cost line is fine — it was never the trim's model

Worth separating, because the line is what I nearly discarded along with the diagnosis.
Predicting each context's winner from `tok_per_fwd / (0.670 + 0.5265·W)`:

| ctx | d1 predicted | d1 measured | d3 predicted | d3 measured | picks |
|---:|---:|---:|---:|---:|---|
| 32 | 1.010 | 1.009 | 0.879 | 0.882 | d1 ✓ |
| 512 | 1.132 | 1.133 | 1.174 | 1.157 | d3 ✓ |
| 1024 | 1.114 | 1.112 | 1.203 | 1.228 | d3 ✓ |
| 2048 | 1.103 | 1.105 | 1.088 | 1.085 | d1 ✓ |
| 4096 | 1.050 | 1.049 | 1.063 | 1.056 | d3 ✓ |

**5/5 correct, and the predicted speedups land within 0.1-2.5%.** The model is sound for
choosing W off-line against a rung table. What it cannot do is be dropped into a function
whose cost is continuous in a different variable.

## Fix

None to the constants. `spec.py` gains a docstring naming the shape mismatch, a comment
saying why repricing is not the fix, and a `__main__` assertion that W=3 and W=4 land on
the same rung at B=4 — the fact that makes the reprice pointless, so it fails if
`LADDER_WIDTHS` changes without revisiting the trim.

The real fix, when it is worth doing, is to make the trim **rung-aware**: enumerate
`LADDER_WIDTHS`, price each with the measured line, and pick the cheapest rung whose
expected tokens beat its cost. That is a different function, not two different constants,
and it only pays if the trim is reached with acceptance near a rung boundary often enough
to matter — which is a distribution nobody has measured yet.

## Rule

**Read the consumer before pricing the producer.** The trim's cost function and the
measurement had the same algebraic form, which is exactly what made the substitution look
safe; 12 lines of `_run_forward` say the consumer pays for `max(len(chains))` rounded to a
rung, and that kills the substitution. Matching forms are not matching models.

Second: **a fix whose effect you cannot state in rows, tokens or milliseconds is not a
fix yet.** "The constants are 13× off" is true and was enough to file a task, but the
question that mattered — *what changes at the shipped batch size* — has the answer
"nothing, then the wrong thing", and it was answerable without a GPU.

Third: **when a correct model gives a wrong answer through a wrong consumer, keep the
model.** The cost line survived this rejection intact and picks the right depth 5/5; only
the plan to inject it into `verify_lens` died.

## Gate

No behavior changed. `uv run python src/tilerl/spec.py` passes with two new assertions,
187 tests pass, ruff clean.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 7491623 | V100 | cuda sm70 | qwen38-27b | rows launched, W=3 vs W=4 at B=4 | **32 vs 32 — a trim between them is free of charge** |
| 2026-09-03 | 7491623 | V100 | cuda sm70 | qwen38-27b | repriced trim's keep≤1 boundary | acceptance **p≈0.92** vs recorded **84.4%** |
| 2026-09-03 | 7491623 | V100 | cuda sm70 | qwen38-27b | cost line picking depth per context | **5/5 correct, 0.1-2.5% on the speedup** |
| 2026-09-03 | 7491623 | V100 | cuda sm70 | qwen38-27b | **#38 reprice verdict** | **rejected — no constants changed** |
