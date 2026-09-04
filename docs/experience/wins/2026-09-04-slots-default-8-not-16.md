# serve --slots defaults to 8, not 16 — 3.5x the context for no lost concurrency, V100, 2026-09-04

> Status: **default flip.** `--slots 16` cost 3.50x the KV pool to buy 8 queue
> positions. 8 is the minimum that keeps `--max-batch 8` reachable, and the
> engine now warns when it is not.

## Context

Fixing the `--blocks 0` OOM
(`errors/2026-09-04-the-fit-spent-the-drafts-own-weight-bytes.md`) made the state
pool's cost visible instead of fatal, and it is large: the fitted KV pool at
`--slots 16` is 757 blocks where `--slots 3` gets 3927. Nothing had priced what the
extra slots buy.

## What Worked

**Read the admission path before measuring, because the two knobs are different
quantities.** A slot is allocated in `submit` (`engine.py:462`) and freed at finish
(`1075`); `max_batch` caps concurrent rows in `_build_plan` (`571`). So:

- `slots` bounds **submitted-but-unfinished** requests — queue depth.
- `max_batch` bounds **concurrent** rows in a tick.
- `slots >= max_batch` is *required*: every admitted row holds one, so below it
  `submit` raises before the batch can ever fill.
- Above `max_batch`, each extra slot is one more request that waits instead of
  getting a 503 (`LinearStatePool exhausted` → `server.py:178` → 503, already
  handled, and the failure path in `submit` frees both blocks and slot).

Measured on the 32 GB V100, 27B with a draft, `--depth 3 --max-batch 8`, one
`build_engine(num_blocks=0)` per process:

| slots | fitted blocks | context | vs 16 | queue depth at max_batch=8 |
|---:|---:|---:|---:|---|
| 3 | 3927 | 62832 tok | 5.19x | *max_batch unreachable* |
| **8** | **2649** | **42384 tok** | **3.50x** | 0 |
| 16 (shipped) | 757 | 12112 tok | 1.00x | 8 |

All three answer a request, and with byte-identical output
(`[10248, 61354, 62290, 44576, 92, 93, 198, 10]`) — the pool size does not move the
tokens, so this is capacity, not quality.

## Decision

**`--slots` defaults to 8.** It is the smallest value that keeps the shipped
`--max-batch 8` reachable, and it buys **3.50x the served context** (12112 → 42384
tokens) for **zero loss of concurrency**. What it gives up is 8 queue positions:
under 9+ simultaneous requests the 9th now gets a 503 instead of waiting. On a
single-card endpoint whose measured B=8 ceiling is already ctx=32
(`errors/2026-09-03-batching-is-non-monotone-padding-rows-cost-3x.md`), 8 queued
requests were never going to be served concurrently anyway — the slots were buying
admission capacity the batch ceiling cannot use.

Not lowered to 3: that makes `--max-batch 8` unreachable, which is a silent
concurrency cap rather than a capacity win.

## Gate

The engine warns when `usable_slots < max_batch`, naming both numbers and the pad
row when the decode graph owns one. **Warn rather than clamp** — `num_slots=2` with
the default `max_batch=8` is a legitimate test config (four already exist in the
suite and now emit it), not a mistake. This is the failure the old default hid in the
other direction: nothing anywhere said that `slots`, not `max_batch`, is the real
concurrency ceiling.

248 passed. The warning fires 4 times, all on small-pool test engines.

## Rule

**When two knobs bound the same thing at different points in the lifecycle, find
which one binds first before tuning either.** `max_batch` reads like the concurrency
limit and is documented as one; the slot allocated in `submit` is what actually binds,
and it binds *earlier* — so a `slots < max_batch` engine caps concurrency with no
diagnostic at all. One grep of the allocation sites answered what a throughput A/B
would have measured around.

Second: **a default is a measurement claim.** `--slots 16` shipped with no number
attached, and cost 3.5x the context of the value derivable from `--max-batch` in one
line of reasoning. Both flags now carry the figure and the flags it was measured at.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-04 | (this) | V100 | cuda sm70 | 27B, draft, depth 3 | slots=8 fitted pool | **2649 blk = 42384 tok** |
| 2026-09-04 | (this) | V100 | cuda sm70 | 27B, draft, depth 3 | slots=16 (old default) | 757 blk = 12112 tok |
| 2026-09-04 | (this) | V100 | cuda sm70 | 27B, draft, depth 3 | slots=3 | 3927 blk = 62832 tok |
| 2026-09-04 | (this) | V100 | cuda sm70 | 27B | context gained by the flip | **3.50x** |

Source: `$HOME/tilerl-logs/fit2.log` (slots=3), `fit3.log` (16), `fit4.log` (8).

## Still open

The **rate** across 12112–62832 tokens is unmeasured (task #57), so this entry claims
capacity only. A larger pool cannot slow a tick down — the KV a tick reads is the
live context, not the pool — but that is an argument, not a measurement.
