# The SSD fetch runs off the tick, above a computed break-even — 2026-09-07

> **Status: pending-remote.** Every operand and every routing decision below is
> measured on the CPU target. The two claims that need a card — that the fetch
> overlaps other sessions' compute, and that the copy's event is waited before
> `insert` publishes — are not, because there is no second stream on `"c"` and a
> local test of either would pass for the wrong reason.
>
> Row 50 PR B. Approach: [`docs/design-ssd-read-path.md`](../../design-ssd-read-path.md).
> Batching prep: [`2026-09-07-load-kv-writes-two-copies-not-two-per-block.md`](2026-09-07-load-kv-writes-two-copies-not-two-per-block.md).

## Context

The read path shipped 2026-09-05 at 1.738–1.821x and had two problems that the
measurement could not see, because it ran one request at a time:

1. it faulted in at whatever length happened to be resident, with no test of
   whether reading beat recomputing;
2. it did the `torch.load` inside `_admit`, inside `step`, under `Engine._lock` —
   so a 1.7 s read stalled every decode row in the batch.

## The break-even is derived, not written down

`(S + n·k)/B < n/R`, so `n* = (S/B) / (1/R − k/B)`. Every operand is read at
runtime, and that is not fastidiousness: **two of them differ per arch by more
than the margin they decide.**

| operand | source | H20 | V100 |
|---|---|---:|---:|
| `k` KV bytes/token | pool dtype × shape | 64 KiB (bf16) | **128 KiB** (f32, `backend.py:353`) |
| `R` prefill tok/s | running mean of this engine's chunks | 2558.6 | ~75 |
| `S` snapshot | one spilled blob, constant at any length | 149.63 MiB | 149.63 MiB |
| `B` tier read rate | the tier's own fetches | 182.6 MiB/s | 182.2 MiB/s |

A constant `k` would be wrong by 2x on one of the two cards; a constant `R`
wrong by 34x. `n*` lands near 16,900 tokens on the H20 and ~73 on the V100 —
same model, same disk, opposite answers.

**An unmeasured tier fetches once and calibrates: a restart is the case the tier
exists for, and there `B` is unknown until a fetch measures it.** This is the one
place the arithmetic had to give way. Refusing on an unknown rate would refuse
every fetch forever, and nothing would ever measure the rate — so `break_even_tokens`
returns 0 rather than the never-fetch sentinel, and judges from the first fetch on.

**`B` is a mean over all fetches, warm ones included — deliberately.** On a restart
the spill is usually still in page cache, so the first fetch calibrates warm: 5.66
GB/s against 0.20 cold on the V100, 28x apart, which collapses `n*` to single
digits. The obvious repairs are wrong for the same reason. A running minimum tracks
the cold rate, but `B`'s job is to predict the *next* fetch, and after a restart the
next fetch is usually warm too — a minimum would refuse reads that would in fact have
been fast. Skipping the first fetch only moves the same bias one sample later, since
every subsequent cached read is warm as well. This is the same concession as the
unmeasured-tier case above, one step on: both accept a `B` that may be wrong in the
permissive direction, because refusing is the expensive error.

What makes the permissive direction safe is what a wrong `B` actually costs.
`lookup` declines a prefix whose fetch is still in flight (`kv_cache.py:1127`) and
returns a miss, so `_admit` proceeds with `matched=0` and the request prefills on the
spot. Nothing ever waits on a fetch. An over-optimistic `B` therefore buys one queued
read on the reader thread — not a slower turn, and not a stalled request. The
deadline at `engine.py:714` is the second bound, not the first; a change that made
requests wait on a fetch would move it to first and this reasoning would need redoing.

`/health` publishes `prefix_break_even_tokens` as `null` rather than the
`NEVER_FETCH` sentinel, so a reader is never handed 2147483648 to compare against a
prompt length.

**And the gate for that first shipped in the shape this entry's own Rule warns
about.** It asserted only that a tier-less engine publishes `null` — an assertion a
stats line hardcoded to `None` also satisfies. Measured, not argued: replacing the
whole expression with a literal `None` leaves the **full suite green, 461 passed**.
Nothing in the tree could tell "publishes null when the sentinel comes back" from
"publishes null always".

The arm that fixes it is the finite one: a store answering with a real `n*` must
reach `/health` as the number, which fails `assert None == 73` on the mutant. So the
Rule below generalises past thresholds — a gate on a *mapping* needs a case on each
side of the map, and the side that reads as "nothing to see" is the one that goes
unwritten.

## Off the tick

`submit` issues the prefetch: roll the hash, probe `resident()`, enqueue. It takes
no blocks and no pool state, so a prefetch nobody collects costs one host buffer.
That is exactly why it can live where the prefix *match* cannot — `engine.py`
already explains the match moved to `_admit` because `submit` has no later tick to
retry an allocation on, and this allocates nothing.

`lookup` **declines** a hit while a fetch is in flight rather than reading the
bytes itself. Doing both would put the whole read back on the tick, which is the
cost the change exists to remove. The deadline in `_build_plan` bounds the wait at
`n/R` from submit; past it the request prefills and the bytes are dropped when
they land.

`load_kv` now permutes *after* the transfer. `.to(cuda)` on a non-contiguous view
materialises a contiguous temp of the whole blob on the host first, so permuting
first pays an extra host copy of every byte. Same-device is a no-op either way —
which is precisely why the test target cannot see this, and why it is written down
rather than tested.

## Four mutations that found real defects

**The caller re-derived the guard.** The walk was bounded below by the break-even
*and* guarded by it. Mutating the guard to `if False` changed nothing: the loop
refused the same lengths a second time, so the guard was dead code and a gate
aimed at it could never fail. One comparison now.

**Two arms could not reach the branch they tested.** Both let the prefetch finish
before calling `lookup`, so the in-flight path never ran — deleting it left them
green. The third arm stalls `torch.load` behind an event and asserts the wait was
counted.

**A test rate made its own operand invisible.** The `k`-comes-from-the-pool gate
compared a bf16 pool against an f32 one at 500 tok/s, where `k/B` is 0.15 µs
against 2000 µs of recompute: both dtypes give the same `n*` and the assertion
could not discriminate. At 2.5e6 tok/s the two terms are the same order and the
2:1 ratio appears.

**A mutation that hit the wrong line.** Deleting `self._abandoned.discard(key)`
left the gates green — because there are two such lines and the one I deleted was
in the error path, not the drop branch. Deleting the branch itself fails two arms.
A mutation is only a control if it lands where you think it did.

Final: six mutations, five red, one green *correctly* (`permute(1,0,2,3,4)` and
`transpose(0,1)` are the same operation on a 5-D tensor, so it was never a
mutation).

## What the bench adds

`scripts/bench_ssd_restart.py` grows a **below-break-even arm** at `n*/2`, whose
pass condition is that it does **not** prefetch. Without it the bench cannot
distinguish "the threshold works" from "fetching is always on": every other arm
sits above `n*`, so a build ignoring the threshold entirely produces identical
rows. It also carries `ssd_tick_loads` out of `/health` — a fault served by a
`torch.load` on the calling thread is the old synchronous path, and on a
single-request arm it shows the same wall clock.

`prefill_rate` and `prefix_break_even_tokens` are additive `/health` keys, so a
live server can be asked why it did or did not prefetch.

## Rule

When a threshold gates an action, the test suite needs a case on the *losing* side
of it. Every arm above the threshold passes whether the threshold exists or not,
so a build that ignores it entirely is indistinguishable from one that honours it
— and the arm that would notice is the one nobody writes, because it is the arm
where nothing happens.
