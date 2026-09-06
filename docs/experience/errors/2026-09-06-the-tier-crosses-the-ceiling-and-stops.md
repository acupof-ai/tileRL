# The snapshot tier crosses the byte ceiling and stops — H20 card 6, 2026-09-06

> Status: negative, structural. `dram_bytes=0` stays the default and now has a
> measurement past the crossover rather than an inference below it.

## Context

The tier's condition on record was `sessions > HBM snapshot budget`, from hit counts:
2 sessions → 0 promotions, 9 → 17, 12 → 24. Every wall-clock arm so far had been
block-bound, so the tier never engaged and the condition was untested above the
crossover. This run put the pool above it: `--max-ctx 49152` (3072 blocks), 12
sessions, 3 turns, real 27B on H20 card 6, `--dram-bytes 4G`.

## Root Cause

**The tier engages for one request and then never again, because a demotion returns
state bytes and zero blocks.**

| | blocks | state bytes | evictions | demotions |
|---|---:|---:|---:|---:|
| turn1 c8 | 91.9% | **99.4%** | 0 | 0 |
| turn1 c9 | 98.0% | 99.4% | 0 | **+7** |
| turn1 c10 | **99.4%** | 94.3% | 21 | 7 |
| final | 84.8% | 76.3% | **262** | 7 |

State bytes reached their ceiling first, by 7.5 points, and 7 demotions fired. Those
demotions moved 5.1 points of state headroom back — and no blocks. The very next insert
found the pool at 99.4% and took `evict_until_free`, which calls `_evict_one` directly
and cannot demote. So did all 262 evictions after it.

**Final account: 262 evictions, 7 demotions, 0 promotions, 12 hits over 351 publishes.
`dram_demote_ms=635` for 7 demotions, 91 ms each, all of it wasted** — nothing was ever
promoted back, so every byte copied to the host was copied for nothing.

The two operands reach their ceilings 1.4 points apart. That is what makes enlarging
the pool useless here rather than merely insufficient: at any pool size where state
binds first, blocks are within a hair of binding too, and demotion relieves only one of
them.

## Two withdrawn numbers

**A "shorter prompts favour the tier" table.** I computed blocks-per-entry from the
prompt length (1411 tokens → 89 blocks) and concluded that short prompts push the ratio
toward bytes. It contradicted my own measurement — the table said 1411 tokens needs
3.38x the pool, the run showed state binding at 0.82x — and the contradiction was the
table's, not the run's. **An entry's blocks are shared and its snapshot is not.** A
session's 4 publishes are nested prefixes sharing blocks by refcount, so measured per
session: 4 publishes, 598 MiB of state (149.5 MiB each, matching the constant-snapshot
finding), but only **86.7 blocks total, not 4 × 89**. Correct operands: the budget holds
29.2 sessions, the pool holds 35.4, state binds — as observed.

**A 40,459-token crossover solved from turn 0.** Right in direction and useless in
practice: at 49,152 tokens state did bind first, for 1 request out of 36.

## Fix

No code change. `dram_bytes` keeps its default of 0, and the flag's help text already
names the cost below the threshold (1.51x worse wall clock). What changes is the
validity criterion for any future arm: **`dram_promotions > 0`, not `dram_demotions >
0`**. This run satisfies demotions > 0 and measured nothing, because a demotion that is
never promoted is pure cost.

The regime that could still pay is one where blocks are structurally cheap relative to
snapshots — many short conversations rather than few long ones — but that is now a
hypothesis with a named operand (blocks per session vs 598 MiB per session), not a
sweep parameter, and this pod's workload is not it.

## Rule

When two ceilings are within a couple of percent of each other, relieving one of them
buys one operation, not a regime. Check the gap between the operands before treating a
threshold as a knob.

And an average over a shared resource is not a per-item cost. Blocks are refcounted
across nested prefixes, so multiplying a per-entry block count by entry count
double-counts every shared page — the error ran toward "the tier has room to work",
which is the direction that goes unchecked.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---|---:|---:|
| 2026-09-06 | 386d7ca | H20 card 6 | cuda | qwen38-27b | n/a | n/a | n/a |

Raw artifacts: `/work/calib12.log` (12 sessions, `--max-ctx 49152`),
`/work/drampress4.log` (2 sessions, 8192, block-bound throughout).

No runtime change, so no perf surface. The 2/8/12 wall-clock sweep is **not run and
should not be**: at 12 sessions the tier produced 0 promotions, so both arms would
measure the same path and any delta would be drift.
