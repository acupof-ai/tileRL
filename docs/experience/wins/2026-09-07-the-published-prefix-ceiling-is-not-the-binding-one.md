# `/health` stated a prefix ceiling 372x too high — 2026-09-07

> Status: Shipped. Measured on the live V100 serve child (pid 2977356, `57b06ea`,
> `--max-batch 1 --max-ctx 32768`), no card taken and no restart.

## Context

`/health` publishes `prefix_capacity`, and an operator sizing a serving workload
reads it as "how many prefixes fit". On the V100 it says **4096**. The store was
holding **11** and evicting on nearly every publish. Both numbers are correct;
only one of them is a ceiling.

`capacity` is the entry-count cap (`PrefixStore.__init__`, default 4096). The cap
that actually binds is `state_bytes`, because a GDN snapshot is a **constant
~150 MiB at every prefix length** — no token axis — so the byte budget divides
into a fixed entry count regardless of how long the prefixes are.

## What Worked

`entries_capacity = min(capacity, state_bytes // snapshot_bytes)`, published as
`prefix_entries_capacity`. Against the live server's own numbers:

| operand | value |
|---|---|
| `prefix_state_bytes_budget` | 1,845,067,776 |
| snapshot, per entry | 156,893,184 |
| derived ceiling | **11** |
| `prefix_capacity` as published | 4096 |
| overstatement | **372.4x** |

The derived value equals the 11 entries the store was actually holding, at 93.5%
budget fill — so the arithmetic is confirmed against an observation rather than
only against itself.

The same reading explains the eviction shape that prompted this: **105 evictions
against 45 blocks freed (0.43 blocks/eviction) with the block pool at 192/2048,
91% free.** Nothing was evicted for want of blocks. Every eviction was state
bytes, on a card whose block pool was almost empty.

## The hit path is not broken, and that is a separate finding

`prefix_hits` read **0** after 115 publishes, which is the same shape as the
`ssd_hits 0` regression another session is chasing. It is not the same defect.
Two identical 86-token requests, back to back on the live server:

| | hits | misses | published | entries |
|---|---:|---:|---:|---:|
| before | 0 | 4 | 116 | 11 |
| after the repeat | **1** | 4 | 116 | 11 |

So the HBM store serves hits on the live path; the 0 was the absence of repeat
traffic, not a broken lookup. Worth recording because a **21-token request moved
`published` by 0** while the 86-token one moved it by 1 — so a probe built from
short prompts measures nothing and reads exactly like a broken store. The
condition is `engine.py:1312`: a publish needs a *decode* step landing on
`materialized % BLOCK_TOKENS == 0`, and `BLOCK_TOKENS` is **16**. At prompt 21
with 4 new tokens `materialized` runs 20→24 and never lands on 32, so there was
no boundary to publish at.

I first wrote that constant down as 64 in this entry, from memory rather than
from `kv_cache.py:22`. The 0 and the 1 are measured; the mechanism was not, until
it was read.

## Rule

A published ceiling must be the binding one. When two caps exist and one is
derived from a byte budget, put the derived number on the wire next to the
declared one — and check it against what the store is holding, since a cap that
agrees with no observation is arithmetic, not a measurement.

## Results

- `prefix_entries_capacity` on `/health`, `entries_capacity` in
  `PrefixStore.stats()` and `NoPrefixStore.stats()`. **Not yet observed on a live
  wire**: the serve child that produced every number above (pid 2977356) predates
  this change, and asking it for the key returns absent. The derivation is proved
  in tests and against that server's *operands*; the key itself lands on the wire
  at the next restart.
- Gate: `test_prefix_state_budget_evicts` grows two asserts — the derived cap
  equals the budget's answer (2 at 1000 B / 400 B), and the fixture's count cap
  must exceed its byte cap or the assert cannot tell a derived value from a
  restated one. Mutant: `entries_capacity = self.capacity` → `assert 4096 == 2`.
- The routing gate `test_every_key_the_store_publishes_reaches_health_or_is_named_as_dropped`
  already covers the new key: dropping the `_build_stats` line fails it at
  `test_kv.py:620`. Confirmed by running that mutant, not by inspection.
- Suite: 459 passed, 14 skipped.
