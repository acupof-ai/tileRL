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

> **Priced 2026-09-08 and it is not worth fixing.** The signature is wrong, and the cost on the
> live path is 2 lost publishes out of 266 unaligned lengths (both at budget 504) — because
> `interior_published == 1` publishes the first interior boundary unconditionally and absorbs
> almost every `last` failure. Two sessions argued the fix independently, on two different
> grounds, and both grounds were wrong:
> [errors/2026-09-08-a-disjunction-reasoned-one-term-at-a-time.md](2026-09-08-a-disjunction-reasoned-one-term-at-a-time.md).

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
guard fixes this. The 30k prompts the bench runs are unaffected, which is why this never showed up in a
measurement.

## Four candidate fixes, all rejected, and the last one for a reason worth keeping

**1. A budget-free predicate.** Scored against the real walk over budgets 2..599 × lengths 2..799, counting
only prompts that had a publishable interior boundary:

| predicate | no spill (default 504–512) | no spill (all budgets) | double spill |
|---|---:|---:|---:|
| today, `x == LB(n)` | 2448 (6.8%) | 37500 (8.2%) | 0 |
| `n - x <= BLOCK_TOKENS` | 4560 (12.8%) | 60652 (13.2%) | 0 |
| `x == ((n-1)//16)*16` | 4560 (12.8%) | 60652 (13.2%) | 0 |
| `n - x <= 2*BLOCK_TOKENS` | 2300 (6.4%) | 30164 (6.6%) | **41962** |

The narrow forms miss twice as often as what shipped; the wide one double-spills. Rejected.

**2. Pass the budget — `LB(n, budget)` replays the walk.** Exact (0 misses, 0 double-spills, every budget) and
cheap: a 30k prefill costs 0.59 ms of Python against 59 GPU forwards. **Rejected as unsound.** The walk depends
on the budget *history*, not the current budget, since `budget` is recomputed every tick and decode rows join
and leave mid-prefill. Sweeping schedules against a constant 512 over lengths 2..5999, the deepest interior
boundary moves on **352–414 lengths** (n=1018: 1008 at a constant 512, 512 under 512/504 alternating). A
replay from 0 cannot see the budgets the earlier chunks used. One length agreed on three schedules, which is
what made this look safe before the sweep.

**3. Retro-spill at DONE.** Publish every interior boundary `spill=False`, remember the deepest, spill it once
the walk finishes. Schedule-independent by construction: **15 unspillable prompts at every realistic budget
against today's 2448**, 0 double-spills. Needs a new `PrefixStore` method, since `insert` refuses a duplicate.

**4. Deferred publish, no store change.** Same idea at the existing call site, suppressed for block-aligned
prompts so `:1076` does not double-spill. Scored exact on every schedule: 0 double-spills, 15 unspillable, 2
publishes per prompt unchanged.

**Rejected, and this is the one worth recording: the snapshot is positional.** `_publish_prefix` clones the
state slot *as it is when called*, and a hit copies it straight back (`engine.py:722`
`self._states.states[slot].copy_(snap_states)`). Publishing position `p` at DONE would pair `tokens[:p]` with
the state of the **whole prompt**. Every future hit on that entry would restore a GDN state from further along
than its tokens, and nothing would raise — the blocks are right, the lengths are consistent, the request
succeeds. Silent wrong inference, which is worse than the silent missing spill it was meant to fix.

So candidate 3's extra store method is not incidental surface: a correct fix has to capture the snapshot at the
boundary and spill it later, which means the store must accept a spill for an entry it already holds. That is
the shape of the real fix, unbuilt.

**That precondition was built, measured, and NOT landed — because the 2448 above is the wrong operand.** It comes
from a chunk-arithmetic replay with no store, no tier and no pressure, so it counts boundary positions that are
*schedulable*, not spills that would *happen*. A boundary entry has to survive to DONE while every other row in
the batch publishes, and it usually does not: `spillable = clamp(snapshot_slots − batch, 0, batch)`, which is
**12.5%** at the V100 default (9 resident snapshots against `max_batch = 8`). Worse, the spill's destination is
the SSD tier, which is under a recorded REJECT on the serve path and defaults off; and for a second-turn hit the
DRAM tier already reaches 100% on its own. So candidate 3 fixes *the boundary position is mispredicted* while the
binding constraint is *the boundary entry does not live to DONE* — capacity, not the publisher. The store-side
interface is kept as a design conclusion, not as code.
[errors/2026-09-08-the-boundary-spill-measured-at-the-wrong-layer.md](2026-09-08-the-boundary-spill-measured-at-the-wrong-layer.md)

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

**One length agreeing on three schedules is not schedule-independence.** The budget-replay fix was checked at
n=30113 against three varying budget schedules, all agreed, and it looked sound. Sweeping 5998 lengths found
352–414 that disagree. A single point cannot distinguish "invariant" from "invariant here".

**A cache entry is a (tokens, state) pair, and a fix that moves one without the other is worse than the bug.**
The deferred publish scored perfectly on every count that mattered — one spill per prompt, every schedule, no
extra publishes — and would have paired each entry's tokens with a later prefix's GDN state. The missing spill
it replaced is a lost optimization; a mispaired snapshot is wrong output that nothing raises on. Score a cache
fix on what it stores, not only on when it fires.
