# There was no depth ceiling: the ladder converges, and the bench measured its warm-up — H20, 2026-09-08

**Date:** 2026-09-08
**Machine:** H20 pod card 0, `qwen38-27b` NVFP4, `bench_chat_interleaved.py --sessions 12 --turns 3
--grow 10 --sys-tokens 30000 --ttft`
**Status:** open — the mechanism is settled, no fix has run, and the fix this entry originally proposed is
withdrawn as a likely regression. Follow-up to
[the #271 regression entry](2026-09-08-four-mechanisms-one-regression-and-a-copied-flag.md), which
established the cost model but attributed the ladder to submission order.

## Context

`prefix_hit_tokens` showed that the k-th prefix hit of a run matches exactly `512 × k` tokens — cumulative
322560 over 35 hits is `512 × 35×36/2` to the token. Six explanations followed for why the depth was
"capped at 20.4% of the prompt". All six were wrong in the same way: **there was no cap.** The run ended
while the number was still climbing.

## The two numbers that settle it

**The conversations share 29759 tokens — 98.8% of the prompt.** Measured on the pod through the server's
own render path (`render_prompt`, `server.py:131`) with the real 27B tokenizer, on token lists, which is
what `lookup` keys on. All 11 pairs, all three turns, gave the same value:

```
turn 0: prompt lens [30113, 30093, ...]   conv 0 vs 1/2/3: LCP 29759 (98.8%), block-aligned 29744
turn 1: 29759 (96.5%)     turn 2: 29759 (93.3%)     min over 12 convs: 29759
```

**And `k` is the hit index, not the session count.** `depth_k = min(512k, 29744)`. The in-band maximum is
reached at `k = 59`. The run had 35 hits, so it stopped at 17920 tokens — 59.5% of the prompt, still
climbing:

| | tokens | % of prompt |
|---|---:|---:|
| deepest hit observed (k=35) | 17920 | 59.5% |
| mean hit | 9216 | 30.6% |
| **the "ceiling" this entry first claimed** | **6144** | **20.4%** |
| in-band max, reached at k=59 | 29744 | 98.8% |

`6144` is `12 × 512`. It came from reading "12 sessions" into an index that counts hits. The ladder
converges on the entire matchable band with no code change and no extra slots; what the bench measured was
a warm-up transient.

## The mechanism, which is real and was never the problem

`interior_published == 1` fires at the first block-aligned chunk end past `pf.prefill_from`
(`engine.py:1050`), and `req.prefill_from = matched` (`:712`). So a row's first interior publish sits one
chunk beyond whatever it matched:

```
row A misses, starts at    0 -> publishes at  512
row B matched  512, starts at  512 -> publishes at 1024
row C matched 1024, starts at 1024 -> publishes at 1536
```

`max_num_batched_tokens` is 512, so the step is the chunk size. Verified: 0 of 35 hits off the `512k`
ladder by more than 300 tokens, `published` exactly 4 on all 36 rows. `prefill_from` counts in **absolute**
token positions (`:1037` `pf.prefill_from += c`, `:1043` `% BLOCK_TOKENS`) — it is seeded from `matched`
but not relative to it.

This is a ratchet with a `+512` step and a fixed point at the in-band maximum. It is slow to warm, not
broken.

## The fix this entry proposed, and why its sign depends on horizon

"K evenly spaced publishes" (`engine.py:1054`). A publisher cannot know the LCP — it sees one request's
tokens — so the only implementable spacing is over `[0, prompt_len]`, putting the K-th rung out of band and
the deepest matchable one at `(K−1)/K`. A **flat** rung is then a cap where the ratchet is a climb, and the
cap is overtaken at a hit count that grows with K:

| K | rung | N=35 | N=58 | N=80 | crossover |
|---:|---:|---:|---:|---:|---:|
| 2 | 15056 | +70.6 s | −6.7 s | −127.2 s | **57** |
| 3 | 20064 | +134.1 s | +99.8 s | +20.4 s | **86** |
| 8 | 26336 | +213.7 s | +233.2 s | +205.2 s | **242** |

(Cumulative matched tokens, flat rung as `r × (N−1)` since the first hit has nothing published, ratchet as
`Σ min(512k, 29744)`, at 373 µs/token.)

So a flat rung is *better the deeper it is* over any horizon short of its own crossover, and K=2 is the
worst choice rather than a free one — it is the only one this fixture's own length would overtake. A first
version of this table reported a single crossover at 58 for every K, by putting K=2's numbers on all three
rows; that made K rungs look like a warm-up optimization with a near-term crossover, which is K=2's
property alone.

Withdrawn with the ceiling: a per-slot ranking that made K=4 the optimum at 3.1x the byte-budget lever, and
a marginal-slot table that made K=2 look free at +8912 tokens. Both computed their gains from 6144.

**A residency question neither model had, and it turns on placement:** `insert` refuses an exact duplicate
(`kv_cache.py:1202-1205` — hash, then `e.tokens == tokens`), so dedup needs two rows publishing the
identical token tuple. A rung at a *fraction* of `prompt_len` does not qualify — 30113/2 → 15056 but
30093/2 → 15040, different tuples — while a rung at an *absolute* index does, since the LCP guarantees
`tokens[0:N]` is identical across all 12 conversations for any `N ≤ 29759`. The absolute grid is not an
arbitrary constant: it is the block-aligned chunk grid the gate already tests (`:1043`
`pf.prefill_from % BLOCK_TOKENS`, with `prefill_from` absolute).

| | per-row residency | with dedup, 12 sessions |
|---|---:|---:|
| K=2 | 36 slots | 2 rungs + 12 decode = **14** |
| K=8 | 108 slots | 8 rungs + 12 decode = **20** |

The run gives no evidence either way: `published 144 = 4 × 36` exactly means all 144 inserts were distinct,
which shows the ratchet never *collides* — `k` advances every hit, so no two hits share a depth — not that
dedup is unreachable. And the benefit is workload-dependent in the way the depth was not: a fixed rung is
shared only while it sits below the LCP, which the publisher cannot know.

**And an open question on the climb rate.** The step is 512 because the gate fires at the first
block-aligned chunk end past `prefill_from`, and the chunk is `max_num_batched_tokens` (`engine.py:197`). A
larger budget makes each advance larger — 2048 reaches the band by hit 15 — for one publish per row either
way. Whether the step tracks the budget exactly is not decidable from this fixture, since a 512-exact ladder
is consistent with both readings; `_pick`'s `_PREFILL_BUCKET` alignment is where it could diverge, the
adopted match also coarsens, and raising the budget costs decode latency on mixed ticks (`:748`
`budget = max_num_batched_tokens - len(decodes)`).

## What the count cap does and does not explain

```
_entries_capacity (kv_cache.py:1431) = state_bytes // snapshot_bytes = 1.0 GiB / 156.9 MiB = 6
```

Every row of every arm reports `ent=5/6` — at the cap continuously. A snapshot is a **constant size at any
prefix length** (`kv_cache.py:1434`), so 6 slots is 6 slots whether they hold 512-token rungs or 30k
entries: the byte budget cannot distinguish shallow from deep, and the surviving entry is chosen by
eviction order alone.

What that costs is the **intra-conversation** deep hit, which is real and separate from the ladder.
Verified on tokens: conv0's turn-0 prompt (30113) is a full prefix of its turn-1 prompt, and its published
entry at 30096 is inside that, so the next turn could match 30096 of 30825. It never does — 0 times in 36
rows. The competing publishes need the *net* count:

```
published    144 / 36 rows = 4 per row
superseded    36 / 36 rows = 1 per row   (engine.py:1362 replaces the decode entry, retiring the previous)
NET RESIDENT                = 3 per row
11 other convs × 3 + 1 own  = 34 slots = 5.21 GiB
```

`--state-bytes`: 1 GiB → 6 slots, 4 GiB → 26 (short), **6 GiB → 39 (first sufficient)**. Two wrong numbers
preceded it — 4 GiB from counting only the prefill publishes, 8 GiB from counting all four published
including the one retired every row. An arm at 4 GiB would fail and read as refuting the lever rather than
underfunding it.

Residency does not depend on `spill`: `insert` (`kv_cache.py:1177-1210`) retains every entry in HBM and
`spill=False` only declines to offer it to the disk tier.

## Three attributions refuted before the ceiling itself fell

**1. A publishing ceiling — "the deep entry is never published."** Refuted by counting: `published` is
144 = 4 per row × 36 rows, one being the `or last` branch at `:1051`. The deep entry is published on every
row. A one-line fix followed and was **priced negative**: measuring the first interior boundary from 0
instead of `prefill_from` makes every row publish at 512 forever — the ratchet's growth is *entirely* due
to each row starting where it matched.

**2. Block-axis retention.** Blocks per entry differ 59x (32 at 512 tokens, 1882 at 30112). **Refuted by a
5.9x pool null:**

| arm | blocks_total | pool peak | evictions | entries/capacity |
|---|---:|---:|---:|---|
| `pub271off` | 48099 | 4245 (**8.8%**) | 139 | 5–6 of 6 on 47/48 rows |
| child arm | 8192 | 3735 (45.6%) | 103 | 5–6 of 6 on 35/36 rows |

139 evictions at 8.8% occupancy is not block pressure. `--blocks` was never the lever.

**3. LRU bootstrapping.** Not needed: the cap is 6 and arithmetic, so a victim exists on every row
regardless of ordering. A fourth candidate cleared: `superseded 36` is 1 per row from the decode path
(`engine.py:1362`), which does not touch the prefill deep entry.

## Rule

**A number taken mid-transient reads as a limit.** Six mechanisms were proposed for a 20.4% ceiling that
did not exist; the ladder was climbing 512 tokens per hit toward 98.8% and the run stopped at hit 35 of
the 59 it needed. Before explaining why a quantity stops at a value, check that it stopped — plot it
against its own index, or extend the run.

**An index's name is a claim.** `6144` is `12 × 512`, and 12 was the session count in the command line. The
cumulative sum `512 × 35×36/2` says the multiplier is the hit count. Every downstream number — five
tables, two per-slot rankings, a free-fix recommendation — was computed from that one substitution.

**A flat replacement for a growing quantity is a cap as well as a floor, and its crossover moves with its
depth.** K=2's rung is overtaken at hit 57, K=3's at 86, K=8's at 242. A fixture shorter than the crossover
cannot see the sign, and a table that reports one crossover for every K hides that the shallowest option is
the worst one.

**A constant-size snapshot means the byte budget is a count, and a count cap is invisible on every axis you
would think to check.** `entries_capacity` printed `ent=5/6` on every row of every arm, by an instrument
added to make it readable, and three explanations reached past it for publish position, block pressure and
eviction recency.

**Read the retire site before counting resident entries.** `published 144 / 36` is 4; `superseded 36 / 36`
is 1. Net is 3, the budget is 6 GiB, and both 4 GiB and 8 GiB came from a published count standing in for
a resident one.

**A simulator that omits the mechanism under test will report that the mechanism does not matter.** The
from-0 policy returned 100% match depth under *both* policies — a 5x contradiction with the card — because
it modelled publishing without eviction.

**Verify a claim's current state before verifying the claim.** Three of four peer confirmations tonight
landed after the thing being confirmed had been withdrawn, and a reconciled two-peer model with a
third-party code check on its load-bearing claim was still wrong, because all three of us were reasoning
above a baseline none of us had derived.
