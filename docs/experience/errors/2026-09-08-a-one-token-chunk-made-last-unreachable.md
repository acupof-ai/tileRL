# A 1-token final chunk made `last` unreachable, and 14 prompt lengths spilled nothing — CPU, 2026-09-08

**Date:** 2026-09-08
**Machine:** local CPU target, tiny model (`tests/test_kv.py::test_last_prefill_boundary_is_a_real_chunk_end`).
No card was available — the V100 held an 8.5-hour serve process — so this is a chunk-arithmetic bug found and
fixed without one; the invariant is exact, not statistical.
**Status:** closed — fix and test in the same commit.

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

## Rule

**A predicate that predicts another function's behaviour needs a test that runs both.** `_last_prefill_boundary`
is nine lines of arithmetic that reimplements the tail handling in `_build_plan`, and the two disagreed on 14
lengths for as long as both existed. Neither is wrong in isolation.

**An alignment added to make something reachable can make a neighbouring case unreachable.** The 64-cut at
`:779` was added because 15 of every 16 prompt lengths had nowhere aligned to publish. It fixed that and
created this, by landing the cursor exactly on a bucket multiple so the remainder chunk computes its own
`short` as zero.

**A silent no-op is the failure mode to look for in a spill path.** Every counter reads normally on these
lengths: `prefix_published` is unchanged, prefill and decode are correct, the request succeeds. The only
observable is a disk tier that never receives an entry, which no assertion covered.
