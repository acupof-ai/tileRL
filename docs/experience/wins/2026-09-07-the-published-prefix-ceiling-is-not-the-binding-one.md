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

`entries_capacity = min(capacity, (state_bytes + dram_budget) // snapshot_bytes)`,
published as `prefix_entries_capacity`. Against the live server's own numbers
(no host tier there, so `dram_budget` is 0):

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

## Correction: the first version ignored the host tier

The derivation as first shipped read `state_bytes // snapshot_bytes`, and that is
wrong wherever a DRAM tier exists. `_demote_one` moves a snapshot to the host and
**leaves the entry matchable** — same tokens, same blocks, still in the index — so
the host's bytes are capacity too. Measured on the config `tests/test_e2e.py`
already uses (`state_bytes=0` with a tier):

| | resident entries | reported capacity |
|---|---:|---:|
| as shipped | 3 | **0** |
| corrected | 3 | 5 |

A capacity **below** the resident count is not a ceiling, and an operator sizing
against it sees a store that cannot hold anything. The V100 numbers above are
unaffected — that server runs no host tier, so `dram_budget` is 0 and 11 stands.

The gate this needed is a second arm, not a tighter assert on the first:
`test_entries_capacity_counts_the_host_tier`, three configs (tier only, HBM only,
both — which must sum). Two mutants, each red on its own assertion: the shipped
`avail = self.state_bytes` gives `(3, 0) == (3, 5)`, and returning `capacity`
unconditionally trips both this arm and the older one. Suite 459 → **460 passed,
14 skipped**.

What let it through: I checked the derivation against the one card in front of me,
which has no host tier, so the operand that was missing was **zero in every number
I looked at**. An arithmetic identity holds on a config where a term is 0 whether
or not the term belongs.

**And the config that would have caught it already existed in the tree.**
`tests/test_e2e.py:869` builds a `PrefixStore(pool, state_bytes=0, dram=dram)` —
the exact shape where the missing term is non-zero. It was one assertion away, in a
file I had already read that day. So the reusable check is not "test more configs";
it is: **before publishing a derived quantity, grep the suite for a fixture where
one of its operands is non-default, and read the derivation against that fixture.**
A term that is zero on the box you are looking at is invisible to every number that
box produces, and the tree usually already contains the case that exposes it.
