# The spec suite's requests ran out of tokens mid-measurement — 2026-08-30

> Status: fixed. Closes the open item in
> [wins/2026-08-29-spec-decode-net-win.md](../wins/2026-08-29-spec-decode-net-win.md):
> "tok/tick is 1.12 at 55.8% acceptance where depth 1 should give ~1.56".

## Context

The speculation verdict rested on a break-even derived from measured
acceptance: `1 + p >= 17.9 / 10.76` needs `p >= 66%` against a measured 55.8%.
But the same run reported **1.12 tokens per tick**, and at depth 1 a tick
commits `1 + n_ok` tokens, so tok/tick must equal `1 + p` = 1.56. Two numbers
from one loop disagreed by 39%, and the entry recorded that and moved on.

## Root Cause

`bench_harness.suite_spec` sized each request at
`max_new_tokens = (ticks + 20) * (1 + depth)`. The loop actually steps
`settle_decode` (up to `4b + 64 + 8b + 40`) + 8 warm + `3 * ticks` timed.

At the defaults (`ticks=20`, B=1, depth 1) that is 80 tokens of budget against
about 109 produced across ~70 ticks. **The request finished around tick 51 of
70**, and the remaining 19 timed ticks generated nothing while still counting
in the denominator:

```
tokens in the timed window / timed ticks  =  (80 - ~16) / 60  =  1.07
```

against 1.12 measured. The `plain` arm never hit it — one token a tick fits
inside 80 — so **only the speculative arm was penalised**.

It also explains the shape the wins entry flagged as unexplained. At depth 2
the budget is 120 and production is ~116: it *just* clears, which is why depth 2
read 1.77 against ~1.89 implied while depth 1 read 1.12 against 1.56. The
defect is depth-1-only because the budget happened to clear at depth 2, not
because anything differs about short chains.

`ms/tick` survived: `_median_windows` takes the MEDIAN of three windows, and
only the last window was mostly empty.

## Fix

`benchkit.SETTLE_BUDGET(b)` names the settle bound that was previously inlined,
and the suite sizes the request from the ticks it will actually run:

```python
budget = (bk.SETTLE_BUDGET(b) + 8 + 3 * ticks + 4) * (1 + depth)
```

The old value was below the worst case in **every** configuration the suite
measures — b=1 d=1 (80 vs 368), b=1 d=2 (120 vs 552), b=8 d=1 (80 vs 536),
b=8 d=4 (200 vs 1340). Depth 2 escaped only because its realised acceptance was
low enough that the worst case did not happen.

Plus a guard, so this cannot come back silently: a row whose running set shrank
during the timed window is printed as void instead of reported.

## What it changes, and what it does not

Correcting tok/tick to `1 + p` raises the speculative ratios and moves no
verdict:

| B | depth | reported | corrected |
|---:|---:|---:|---:|
| 1 | 1 | 0.57x | **0.79x** |
| 8 | 1 | 0.15x | **0.22x** |

Speculation still loses at every point measured, and the break-even (`p >= 66%`
against 55.8%) is unaffected — it was derived from ms/tick and acceptance, both
of which were sound.

## Rule

When two numbers out of ONE loop are inconsistent, that is a measurement bug
until proven otherwise — do not record it as an open curiosity and reason past
it. The inconsistency here sat in the same table as the verdict it qualified.

Corollary: a benchmark's request budget is a measurement parameter, not a
detail. Derive it from the ticks the loop will run, and assert that nothing
finished before the last one.
