# A 1-token final chunk made `last` unreachable, and the fix only holds at one budget — CPU, 2026-09-08

**Date:** 2026-09-08
**Machine:** local CPU target, tiny model (`tests/test_kv.py::test_last_prefill_boundary_is_a_real_chunk_end`).
No card was available — the V100 held an 8.5-hour serve process — so this is chunk arithmetic found and
fixed without one; the invariant is exact, not statistical.
**Status:** **open** — the shipped fix (c14511b) closes the case at `budget == 512` and the bug returns as soon
as a decode row shares the tick. The real defect is `_last_prefill_boundary`'s signature; see the last section.

## Context

`_last_prefill_boundary(n)` (`engine.py:57`) predicts where `_build_plan` ends the final prefill chunk, and
`_finish_prefills` compares against it to decide `spill=` — the one publish per prompt that reaches the disk
tier (`engine.py:1045`, `:1057`). Nothing checks that the prediction is a position the planner actually stops
at. Reading the boundary helper while writing up an unrelated depth finding, the question was whether it can
disagree with `_build_plan`. It can, on 14 lengths under 4000.

## The failure

Replaying `_build_plan`'s chunk arithmetic over every `n` in 2..4000:

```
n=  65  lb=  48  chunks end at [64, 65]
n= 129  lb= 112  chunks end at [128, 129]
n= 513  lb= 496  chunks end at [512, 513]
n=1025  lb=1008  chunks end at [512, 1024, 1025]
```

14 lengths total, one per `_PREFILL_BUCKET × k + 1` and per `max_num_batched_tokens × k + 1`. On each, the
helper names a position no chunk ends at, so `last` is never true. Replaying the publish gate confirms the
consequence:

| n | publishes (position, spill) | reaches disk? |
|---:|---|---|
| 64 | (64, True) | yes |
| **65** | (64, False) | **no — nothing** |
| 80 | (64, False), (80, True) | yes |
| **1025** | (512, False) | **no — nothing** |
| 30113 | (512, False), (30096, True) | yes |

No error, no counter, no log line. The prompt prefills correctly and serves correctly; only the spill silently
does not happen.

## Root cause: the back-off guards the wrong chunk

`engine.py:786` cuts a ragged tail so the prompt-complete publish lands on a block boundary, and its `tail == 1`
branch exists because "a 1-token tail reaches the kernels with a zero block size". But it fires on the chunk
that **carries** the tail:

```
n=65:  first chunk aligns 65 -> 64  (the _PREFILL_BUCKET cut at :779)
       so the tail is a chunk of its OWN: at=64, chunk=1
       end == n, tail == 1, but short = (65//16)*16 - 64 = 64 - 64 = 0
       `short > 0` is False -> the back-off never runs -> the 1-token chunk ships
```

The 64-alignment that exists to make a publish point reachable is what creates the case: it lands the cursor
exactly on a bucket multiple, leaving the remainder alone in the next chunk, where `short` computes to zero
against its own start.

## Fix

Give up a block when the *current* chunk would leave a 1-token remainder, before the tail logic runs:

```python
if len(r.tokens) - (r.prefill_from + chunk) == 1 and chunk > BLOCK_TOKENS:
    chunk -= BLOCK_TOKENS
```

Costs nothing: over n=2..4000 the planner emits **21606 chunks before and 21606 after** — the 17-token tail
replaces a 16+1 pair rather than adding a forward. 1-token final chunks go 14 → 0.

## The test, and its negative control

`test_last_prefill_boundary_is_a_real_chunk_end` drives the real `_build_plan` at n ∈ {65, 129, 513, 1025},
collects the chunk ends, and asserts both that no chunk is 1 token and that `_last_prefill_boundary(n)` is
among them. Verified red with the fix removed — all four fail, on the 1-token assertion at the position the
probe predicted (`1-token chunk at 1024`).

## The fix holds at one budget only, and the scope claim was wrong

The commit says the fix costs no extra forward and takes 1-token chunks 14 → 0. Both are true **at
`budget == 512`**. `budget` is `max_num_batched_tokens - len(decodes)` (`engine.py:752`), so any decode row
sharing the tick lowers it — the common serve case — and the bug returns:

| budget | 1-token chunks | `lb` not a chunk end |
|---:|---:|---:|
| 512 | 0 | 0 |
| 511 | 15 | 16 |
| 510 | 0 | 36 |
| 508 | 0 | 31 |
| 504 (default `max_batch=8`) | 0 | 28 |

Over the whole default-config window (budget 504–512, n 2..4000): **298 lengths still miss and 29 still ship
a 1-token chunk.** The first report of this fix said the invariant was exact; it is exact for one value of a
variable I had held fixed without noticing it was a variable.

A widened guard was tried and reverted. `chunk -= min(BLOCK_TOKENS, chunk - 1)` covers budgets at or below
`BLOCK_TOKENS`, but it fires on *mid*-prefill chunks too, where subtracting produces the 1-token chunk itself
(measured at n=33/budget=16: `at=16, chunk=1`). Swept over budgets 2..599 × lengths 2..799 counting **every**
chunk rather than only the last: no-guard 7857 one-token chunks, flat-block guard 3299, widened guard 3299.
It changes nothing and adds a case.

## The real defect is the signature

```
_last_prefill_boundary(n)                       -> one argument
the actual last interior boundary = f(n, budget) -> two
```

| n | `LB(n)` | b=512 | b=511 | b=508 | b=504 |
|---:|---:|---:|---:|---:|---:|
| 961 | 944 | 944 | **448** | **448** | **448** |
| 1473 | 1456 | 1456 | **448** | **448** | 1456 |
| 1985 | 1968 | 1968 | **448** | **448** | 1968 |
| 30113 | 30096 | 30096 | 30096 | 30096 | 30096 |

A ragged budget (not a multiple of `_PREFILL_BUCKET`) leaves the cursor off-bucket after the first chunk, and
every later chunk end shifts with it. One argument cannot express a two-argument function, so no per-chunk
guard fixes this — the helper has to take the budget, or `_finish_prefills` has to stop predicting and decide
`spill` from the remaining length it already has. The 30k prompts the bench runs are unaffected, which is why
this never showed up in a measurement.

## Rule

**A predicate that predicts another function's behaviour needs a test that runs both.** `_last_prefill_boundary`
is nine lines of arithmetic that reimplements the tail handling in `_build_plan`, and the two disagreed for as
long as both existed. Neither is wrong in isolation.

**An alignment added to make something reachable can make a neighbouring case unreachable.** The 64-cut at
`:779` was added because 15 of every 16 prompt lengths had nowhere aligned to publish. It fixed that and
created this, by landing the cursor exactly on a bucket multiple so the remainder chunk computes its own
`short` as zero.

**A silent no-op is the failure mode to look for in a spill path.** Every counter reads normally on these
lengths: `prefix_published` is unchanged, prefill and decode are correct, the request succeeds. The only
observable is a disk tier that never receives an entry, which no assertion covered.

**Sweep the variable you held fixed before reporting an invariant.** The 21606-chunks-either-way figure and the
14 → 0 count were computed at `budget=512` and reported as properties of the fix. `budget` is a per-tick
quantity; one decode row makes it 511 and the bug is back. The probe was written from the same reading of the
code as the fix, so it could not have disagreed with it.

**Counting only the case you are fixing hides the case you are creating.** The first sweep checked whether the
*final* chunk was 1 token, because that was the bug. The widened guard produces 1-token chunks *mid*-prefill,
and three parametrized tests caught in one run what 1198 swept lengths had missed.
