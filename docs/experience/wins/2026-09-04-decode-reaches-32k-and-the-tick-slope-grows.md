# Decode reaches 32K on one V100 at 15.3 tok/s, and the tick slope grows — sm70, 2026-09-04

> Status: measurement (Task #57). The fitted pool's newly reachable contexts are
> served: **8192 / 16384 / 32768 at 38.0 / 26.7 / 15.3 tok/s**. The tick is the whole
> curve and its **marginal cost per 1K grows 2.20 → 3.83 ms**, so context is not a
> linear cost here.

## Context

`--slots 8` opened a 42384-token pool
([wins/2026-09-04-slots-default-8-not-16.md](2026-09-04-slots-default-8-not-16.md)),
and that entry claimed capacity only — "a larger pool cannot slow a tick down" was an
argument, not a measurement. This measures the rate across what the pool reaches.

**Scope corrected from the task's "8K-64K":** at `--slots 8` the pool is 42384
tokens, so ctx=65536 + 128 generated does not fit. `submit` rejects it at
`engine.py:456` rather than OOMing, so a run there would measure a 400, not a rate.

## What Worked

One process, one pool (4146 blocks, 8810 MiB, sized for 32768), `--tokens 128`,
B=1, depth 3:

| ctx | tok/s | ms/tok | tick ms | Δ tick per 1K |
|---:|---:|---:|---:|---:|
| 8192 | 38.0 | 26.3 | 76.0 | — |
| 16384 | 26.7 | 37.5 | 101.2 | **3.08** |
| 32768 | 15.3 | 65.3 | 162.6 | **3.83** |

With the 09-03 rows (`--tokens 64`, so tick is comparable and acceptance is not):

| segment | Δ tick per 1K |
|---|---:|
| 1024 → 2048 | 9.16 |
| 2048 → 4096 | 2.20 |
| 4096 → 8192 | 2.52 |
| 8192 → 16384 | 3.08 |
| 16384 → 32768 | 3.83 |

**The tick is monotone and its slope is not.** Past the 2048 step (attributed
separately in task #59) the marginal cost per 1K of context rises **2.20 → 3.83 ms,
1.74x**. Over 8192→32768 the tick grows 2.14x against 4x the context: sub-linear in
total, super-constant at the margin.

**32K is served on a 32 GB card at a two-digit rate**, which is the capacity claim
the slots entry could not make.

## What this does NOT measure

**The acceptance column is not a curve and is omitted deliberately.** tok/forward
depends on `--tokens` — the measurement window closes at the first completion — and
the 09-03 rows are 64 while these are 128
([errors/2026-09-04-four-candidates-cleared-for-a-flag-difference.md](../errors/2026-09-04-four-candidates-cleared-for-a-flag-difference.md)).
Within these three rows, all at 128, acceptance falls 2.89 → 2.70 → 2.49 (0.862x
over 4x context), which is consistent with the 09-03 finding that acceptance is
roughly flat in context; but three points at one window are not enough to publish a
context law, and the cross-window comparison is meaningless.

## The mechanism the slope implicates, and one reading I withdrew

sm70's split attention keys its split count on **query width only**
(`registry.py:23`, `sm70_kvsplit(s)` → 32 below S=8, 16 at or above). History never
enters, so from 1K to 32K the count is fixed at 32 and each split's serial tile count
grows: `per = ceildiv(n, KVSPLIT)` at `block_N = 16` gives **16 serial tiles at 8192
and 32 at 16384**.

**I withdrew "so make the split count history-aware".** The grid is
`T.Kernel(KVSPLIT, S*H, B)` where `H = 24` is the *query* head count, not `Hkv = 4`.
At S=4 that is `32 × 4 × 24 = 3072` blocks on 80 SMs — **38.4 blocks/SM**, already
saturated at ctx=1024. More splits do not expose more parallelism; they subdivide
the same work and lengthen the combine kernel's `T.unroll(KVSPLIT)`. Occupancy is not
what binds, so the grid arithmetic does not license the fix I was about to propose.

Attributing the 25.3 ms that 8192→16384 costs needs `prof_decode_budget.py --ctx`
by kernel class, not a derivation from the grid.

## Rule

**A capacity claim and a rate claim are different measurements, and the argument
that bridges them is not one.** "A larger pool cannot slow a tick down — the KV a
tick reads is the live context, not the pool" is correct and it still left the rate
across 12112–62832 tokens unknown, including whether the *slope* is constant. It is
not: 1.74x across the range measured.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-04 | b20ca9b | V100 | cuda sm70 | 27B d3 B=1 | decode @ctx 8192 | **38.0 tok/s** |
| 2026-09-04 | b20ca9b | V100 | cuda sm70 | 27B d3 B=1 | decode @ctx 16384 | **26.7 tok/s** |
| 2026-09-04 | b20ca9b | V100 | cuda sm70 | 27B d3 B=1 | decode @ctx 32768 | **15.3 tok/s** |
| 2026-09-04 | b20ca9b | V100 | cuda sm70 | 27B d3 B=1 | tick marginal, 2048→4096 | 2.20 ms/1K |
| 2026-09-04 | b20ca9b | V100 | cuda sm70 | 27B d3 B=1 | tick marginal, 16384→32768 | **3.83 ms/1K** |

Source: `$HOME/tilerl-logs/lc18.log`. The 1024–4096 rows are `lcfix2.log`
(`--tokens 64`); only their tick column is used here.

## Still open

Two things. The slope's growth is unattributed — it is the split kernel's serial
depth by arithmetic, but that is a hypothesis and the profiler has not run at these
contexts. And ctx=65536 is unreachable at `--slots 8`; it needs `--slots 3` (62832
tokens) or an f16 pool (task #56 halves bytes/token), neither measured.
