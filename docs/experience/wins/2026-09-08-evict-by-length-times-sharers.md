# Evict by prefix length × sharers: +42% prefix reuse, and five refuted policies — CPU, 2026-09-08

**Date:** 2026-09-08
**Machine:** local CPU target (`TILERL_TARGET=cpu`), tiny model
**Cell:** 6 sessions × 2048 tokens × 2 turns through the engine's `submit`/`step`,
`state_bytes` pressure, no tier
**Verdict:** accepted for the cross-session case; the single-conversation case is
untouched and stays open

## Context

[A miss self-reinforces](../errors/2026-09-07-a-miss-self-reinforces.md) attributed
the H20's 199.35 s / 1-of-12-hits cell: a miss prefills from token 0 and publishes at
every chunk end, 62 publishes at a 31k prompt, into a 6-snapshot budget. Pure LRU
then evicts the shared head first — it is the least recent entry in the family — so
the next session misses and floods the store the same way.

This entry is the eviction-policy half. Six policies were measured; one is accepted,
five are refuted, and the sixth refutation is the shape of the remaining defect.

## The instrument, and the reading that was vacuous first

`/tmp/probe2.py`: 6 sessions, 2048-token prompts, two turns, driven through
`eng.submit` / `eng.step` so the publishes come from the engine's own chunk
boundaries rather than a hand-written `insert`. The score is **reuse in tokens**, read
from the store's lookup at each turn-2 request.

The first version read `_prefix.lookup(tokens)` **after** the session finished. That
is vacuous: by then the row's own publishes are in the store, so both arms score a
hit the row itself had just created. Under that reading pure LRU appeared to reuse
2064 tokens per session and the fix 512 — the fix looked refuted, and the number was
the instrument's.

Reading the **first** lookup of each request (admission) separates them. Same grid,
same code, the two policies then differ by 42%. A hit is only a hit if it existed
before the request that used it.

Grid: shared-head length 0 / 512 / 1024 / 1536 tokens × budget 4 / 6 / 8 / 12
snapshots. 2048 tokens reusable per session, 12288 per cell, 196608 per grid.

## Results

Reuse in tokens at admission, turn 2, summed over 6 sessions:

| shared, budget | LRU | (e) | (e′) | (e′)+reparent | (f) | **(g)** |
|---|---:|---:|---:|---:|---:|---:|
| 0, 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0, 6 | 0 | 3072 | 3072 | 3072 | 3072 | 2048 |
| 0, 8 | 0 | 3072 | 5632 | 3072 | 3072 | 6144 |
| 0, 12 | 0 | 6144 | 9216 | 3072 | 12288 | **12288** |
| 512, 4 | 512 | 3072 | 6144 | 3072 | 3072 | 0 |
| 512, 6 | 3072 | 3072 | 8192 | 3072 | 3072 | 3072 |
| 512, 8 | 3072 | 6144 | 9216 | 3072 | 12288 | 6144 |
| 512, 12 | 3072 | 6144 | 9216 | 3072 | 12288 | **12288** |
| 1024, 4 | 6144 | 6144 | 6656 | 3584 | 3072 | 6144 |
| 1024, 6 | 6144 | 6144 | 9728 | 6144 | 3072 | 6144 |
| 1024, 8 | 6144 | 9216 | 11776 | 6144 | 12288 | 8192 |
| 1024, 12 | 12288 | 9216 | 11776 | 6144 | 12288 | 12288 |
| 1536, 4 | 9216 | 9216 | 9216 | 9216 | 3072 | 9216 |
| 1536, 6 | 9216 | 9216 | 9216 | 9216 | 3072 | 9216 |
| 1536, 8 | 10240 | 9216 | 9728 | 9728 | 12288 | 10240 |
| 1536, 12 | 12288 | 12288 | 11776 | 12288 | 12288 | 12288 |
| **total** | **81408** | 101376 | 130560 | 83968 | 110592 | **115712** |
| cells worse than LRU | 0 | 2 | 3 | 3 | 4 | **1** |

- **LRU** (baseline). The whole `shared 0` column is 0 at every budget from 4 to 12:
  with no shared head there is nothing for recency to protect, and LRU loses
  everything. That column is the cascade.
- **(e)** extensions before heads, longest extension first. 101376, loses 2 cells. It
  evicts a session's own longest entry even when the budget could hold it.
- **(e′)** the same but LRU among extensions. 130560 — the best number in the table
  and **not real**; see the trap below.
- **(e′)+reparent**, the honest form of (e′): 83968, 2560 tokens above LRU across 16
  cells. Refuted.
- **(f)** middles before leaves. 110592 but loses 4 cells, all small-budget with a
  long head.
- **(g)** value = `len(tokens) × sharers`, evict the minimum. **115712, +42% over
  LRU**, one cell worse: `(512, 4)`, 0 vs 512, a 4-snapshot budget over 6 sessions
  where nothing survives either way.

## The trap: (e′)'s gain was a bug, not a policy

(e′) protected an entry when `entry.parent is not None and entry.parent in
self._by_id`. Dropping a **middle** entry left its children pointing at a dead eid,
so `parent in self._by_id` read False, the orphans read as family heads — and the
policy protected them. That accidental protection was the entire gain: closing the
chain in `_drop` took 130560 down to 83968, cell for cell (`(0,12)` 9216→3072,
`(512,8)` 9216→3072, `(1024,8)` 11776→6144).

(e) never hit it because longest-first always evicts a leaf. The bug was found by a
gate at a 2-entry budget over **two** families: `resident [(2, 48, 1), (8, 16, None)]`
with family 1's true head gone.

What the accident was doing is the finding. It protected the **deepest** resident
prefixes, not the heads — and a request matches the longest stored prefix, so the
deepest entry is the only one that saves its row tokens. Protecting heads protects
the entry that saves the least.

## Why the weight is a product

`sharers` starts at 1, and rises on a lookup match and on a dedup re-publish by
another row. Both are events the store already sees, so it costs no walk — unlike
(e)'s `_longest_resident_prefix`, which walked the prompt at every insert.

Each factor alone is worse, and the gate in `tests/test_kv.py` kills all three
degenerate forms. Mutating `_evict_one` and running that one test:

| `_evict_one` key | gate |
|---|---|
| `next(iter(self._by_id))` (pure LRU) | RED |
| `len(e.tokens)` | RED |
| `e.sharers` | RED |
| `len(e.tokens) * e.sharers` | green |

The gate's first version went **GREEN under sharers-only** — vacuous. It needed a
second arm: a 16-token prefix matched 3 times (48) must lose to a 128-token one
nobody has matched (128), because what a hit saves is tokens, not matches. It also
needed two matches rather than one, because at one match a 64-token head scores
64×2 = 128 and **ties** a private 128-token leaf; a tie falls to dict order, where the
head is older, so a one-match fixture reads as a loss for reasons unrelated to the
policy.

## Refuted before these

- **REPLACE at the prefill publish site**, mirroring the decode site's fix: 5 hits →
  0. A prefill chunk's entry is what a *later* request matches against, so retiring
  the previous chunk destroys cross-session sharing.
- **Hit-aware eviction**: a no-op. At eviction time every entry has `hits == 0` — the
  publishes that cause the pressure have not been matched yet, by construction.

## The limit, stated plainly

(g) fixes the **cross-session** case and does nothing for the single-conversation
one. Within one conversation nothing has been matched yet, so every entry has
`sharers == 1`, length alone decides, and the head is the shortest entry — it goes
first. `tests/test_kv.py::test_a_prompts_own_publishes_evict_the_prefix_it_shares`
is `xfail(strict)` for exactly this, with that reason.

That cell needs the **publish count** cut, not a better victim: 62 publishes from one
31k-token prompt exceeds any pressured budget before eviction policy has a choice to
make. Separate PR, after the fp8 KV work.

## Rule

**A cache hit only counts if it existed before the request that used it.** Score a
prefix policy at admission, never after the request finishes — a row's own publishes
land during the request and make every policy look equal.

**An eviction policy that reads a pointer between entries must maintain that pointer
in the one place entries are removed.** (e′)'s 130560 was orphaned parent ids reading
as roots; the policy looked 60% better than its honest form.

**Rank a cached prefix by what it saves, which is tokens × rows.** Either factor
alone is measurably worse, and a gate that does not kill both degenerate forms has
not tested the product.
