# Four mechanisms for one regression, all refuted, and the variable was a flag I copied — H20 + CPU, 2026-09-08

**Date:** 2026-09-08
**Machine:** H20 pod card 0 (cells), local CPU target (diagnosis)
**Status:** open — the cross-commit difference at a fixed pool is unexplained. The confound is found.

## Context

[#271](../wins/2026-09-08-cut-the-prefill-publish-flood.md) cut prefill publishing from every interior
chunk boundary (62 entries per 31k prompt) to the first interior boundary plus the last (2 per row).
Accepted on a CPU token-reuse grid: 104448 vs pure LRU's 81408, +28%.

Re-running [the 09-07 DRAM tier cell](../wins/2026-09-07-the-dram-tier-is-357x-when-the-budget-is-pressured.md)
at its exact parameters to settle that entry's Pending flag produced 403.01 s against its 199.35 s,
and the tier that had been worth 3.57x there promoted nothing. Four mechanisms were proposed for that
over the next two hours. All four were refuted, each because it was confirmed against a different
operand than the one the card ran.

## The data that settles it

`final_stats` from all four cells, same sha (a43a379), same publisher:

| cell | published | superseded | evictions | demote | promote | blocks_total | pool_peak | occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cell357off | 144 | 36 | **103** | 0 | 0 | 8192 | 3735 | 45.6% |
| cell357on | 144 | 36 | 86 | **103** | **0** | 8192 | 8118 | **99.1%** |
| pub271off | 190 | 46 | **139** | 0 | 0 | 48099 | 4245 | **8.8%** |
| pub271on | 190 | 46 | **0** | **175** | **36** | 48099 | 30865 | 64.2% |

Two readings of this table were wrong before the right one. First, that pool size explains the
evictions — refuted by `pub271off`, which evicts the most of any arm at **8.8% occupancy**. Second,
that "the cell ran at 99.1% occupancy" — that is the tier-**on** arm; the off arm at the same pool
and workload peaks at 45.6%.

### The identity: one event stream, two dispositions

```
cell357off evictions 103  ==  cell357on demotions 103         exact
pub271off  evictions 139  vs  pub271on demotions 175 = 139 + 36,  promotions = 36
```

The byte-pressure event stream is 103 in cell357 and 139 in pub271, **the same in both arms of each
pair**. What differs is the disposition, selected by `self._dram is not None` in the eviction loop:
with no tier each event evicts, with a tier each event demotes. The +36 is re-demotions — `_demote_one`
(`:1400`) skips entries already demoted, and `promote` clears the flag, so a promoted entry becomes
eligible again.

### The mechanism: the tier converts byte pressure into block pressure

Demotion frees snapshot bytes and **keeps the entry**, which keeps its blocks alive — the documented
purpose of `_demote_one`. So enabling the tier moves the same pressure from the byte axis to the block
axis. cell357 goes **45.6% → 99.1%** peak occupancy purely by enabling it, same pool, same workload.

At 99.1% the block path then forces 86 evictions, each of which drops a demoted entry and calls
`_dram.forget` (`:1365`), orphaning the host copy. Promotion is reachable only from `lookup` on an
entry still in the index (`:1252`), so those snapshots can never be fetched: **103 demotions and
1864 ms of `dram_demote_ms` for 0 promotions**, with 5.9 GiB of tier budget unused. `pub271on` is the
same conversion landing at 64.2% — under the ceiling, so entries survive and 36 promotions fire.

### The locality: `evict_until_free` is the one eviction path with no demote branch

The byte loop at `:1225` was deliberately taught to prefer demotion over eviction. The block path was
not:

```
engine.py:682    if self._kv.free_blocks + self._prefix.reclaimable_blocks() < needed:
engine.py:684        self._prefix.evict_until_free(needed)
engine.py:950        self._prefix.evict_until_free(growth)

kv_cache.py:1330  def evict_until_free(self, blocks: int) -> None:
                      while self._pool.free_blocks < blocks and self._by_id:
                          self._evict_one()
```

No `_demote_one`, no guard, straight to `_evict_one` → `_drop` → `_dram.forget`. So the tier's own
block retention drives pressure into the single code path that cannot use the tier.

**And that asymmetry is not an oversight.** At `:684` the caller needs `needed` blocks *now* to admit a
row, and demotion frees **zero** blocks — it frees snapshot bytes and keeps the blocks. There is nothing
a demote branch could do there; the entry has to go. Demotion is a byte-axis tool, and the block path is
where blocks bind. So the fix is not "teach the block path to demote": it is retain fewer blocks, or
size the pool against the tier's retention.

`reclaimable_blocks` (`:1334`) looks like the number that would separate "these evictions were
unavoidable" from "the accounting double-counts shared blocks", and the admission check at `:682`
already consults it. **It cannot answer that question.** It counts a block when
`refcount[b] == n`, where `n` is the store's own hold count *across all its entries* — so a block held
by two nested entries of the same row counts as reclaimable by construction. It separates
store-held from outside-held, not store-only from sibling-shared.

What the cells do show is a flat per-eviction yield:

| arm | blocks_freed / evictions | pool peak |
|---|---:|---:|
| cell357off | 46578 / 103 = **452** | 45.6% |
| cell357on | 41749 / 86 = **485** | 99.1% |
| pub271off | 69842 / 139 = **502** | **8.8%** |

452 / 485 / 502 across three pools and two dominant eviction paths, ±5%. That flatness is the
robust observation, and it most likely reflects **nesting geometry fixed by the publisher** rather
than anything about pool pressure.

**The yield settles nothing about who holds the blocks.** Three readings of this one number were
proposed and all three withdrawn, and the reason is instructive: the number never changed, only the
population we each thought we were averaging over.

1. *Store-only holds dominate* — because 485 is a large fraction of a row. Withdrawn: `_drop` frees by
   refcount decrement, so 485 of 1926 means 74.8% did **not** come back.
2. *So an outside holder retains that 74.8%, i.e. a live slot* — because under first+last a row's
   family is `{32, 1926}`, and dropping the deep entry while the shallow sibling lives should free
   1894, not 485. Withdrawn: the premise is wrong. cell357 published **144** entries over 36
   session-turns = **4.0 per session-turn**, not 2 — first+last contributes 2 and the rest are
   decode-boundary publishes (`engine.py:1355-1358`), with `retire` dropping only the immediately
   previous one (`superseded 36` = one per session-turn). So a family is a nested chain of ~3 live
   entries, not a pair.
3. *Therefore the eviction mix explains it* — LRU evicts across all entries, and a family's shallower
   members free ≈0 while a deeper member lives:

   | share of evictions that drop the family's deepest member | expected yield |
   |---:|---:|
   | 100% | 1894 |
   | 50% | 947 |
   | **26%** | **492** (measured 485) |

   No outside holder is required. But this is *consistent with* the data, not shown by it: a mix with
   no live holder and uniform deep drops with a live holder on 75% produce the same average.

So the yield average cannot discriminate, and the flatness has a deflationary reading too — if the
eviction mix follows the publisher's family geometry, and the publisher is identical in all four arms,
~25% is expected regardless of pool or pressure path. That makes 452/485/502 an artifact of the
publisher rather than an invariant across eviction regimes.

The line that would settle it: per eviction, the count of the entry's blocks where `refcount[b] > n`
for the store's own hold count `n`, **logged with the entry's token length** — the length separates
last-of-family drops from shallow ones, which the refcount count alone cannot. Not logged; the split
stands unmeasured.

`pub271off` remains decisive for the one thing it was used for: 139 evictions at **8.8%** pool
occupancy cannot be capacity, and those are byte-path evictions with the block path never firing. That
does not depend on the yield.

**So the pool size is the threshold, not the cause, and `cell357on` is net negative against
`cell357off`**: worse block occupancy, zero promotions, real demotion cost — caused by the tier working
exactly as specified. The unstated precondition is that **the tier is a win only while its own block
retention stays under the pool ceiling**, and the 09-07 cell that licensed 3.57x satisfied that by
accident of configuration.

## The four refuted mechanisms

Each was reported, some to a peer's summary to the user, before being withdrawn.

**1. A constant 512-token match depth.** A probe measured `_match_prefix` returning 512 tokens at
every prompt length and this was reported as the card's mechanism. But that probe used a **partial
sharer** diverging 64 tokens before the end, for which 512 is correct behaviour — the deep entry's key
covers the full prompt, so a diverging follower cannot match it. The card's turn 1 is a
**continuation**, which shares everything; measured on that shape it matches **98.1%**. The publish
gate is `interior_published == 1 or last`, and `last` does fire: `_last_prefill_boundary(30826)` is
30816 and `_pick` cuts the final chunk to a block boundary so `prefill_from` lands there. The deep
entry at 99.97% is published and found.

**2. The tier is inert under count pressure.** `kv_cache.py:1225-1235` guards demotion with
`len(self._by_id) <= self.capacity` inside a `while len > capacity or bytes > budget` loop, so under
**count** pressure the guard is false and the entry is evicted instead of demoted. True, and not the
card: the card's "budget 6" is `state_bytes` ÷ 157 MiB with `capacity` at its 4096 default and 24
entries, so it enters the loop through the **byte** term with the guard satisfied. Refuted in one line
by the card's own `demotions 103` — a count-inert tier cannot demote 103 times. (The guard remains a
latent defect for a regime nothing has run.)

**3. The deep entry is evicted by its own row's shallow sibling.** Reproduced at `capacity=6`:
the store held `[(512,'R'),(1024,'R'),(512,'R'),…]` and a continuation matched 0.0%. But `capacity=6`
is the count regime from (2), which the card is not in. Driven in the card's regime — byte pressure at
capacity 4096, synthetic non-empty snapshots so `_snapshot_bytes` is real — **evictions are 0** and
deep hits are 12/12.

**4. Block-axis retention.** The sibling is 32 blocks against the deep entry's 1926, and nested
prefixes **share blocks by refcount**: measured, a store entry over a live row's blocks costs **0**
extra blocks (free_blocks unchanged, refcount 2 while live, 1 after the row releases). So every
publisher's retained block set is just its deepest entry — flood 1920, first+last 1926, **6 blocks
apart**. The block axis cannot distinguish publishers.

A fifth hypothesis, that the eviction difference was an accounting shift into `superseded`, is
refuted by the table: `superseded` is 36 in both cell357 arms and 46 in both of the others. Flat.

## Resolved: 2.03x at one variable, and the mechanism is hit depth

`169d7bd` (pre-fix) run at cell357's exact flags — same workload, same `--blocks 8192`, same
`--state-bytes`, tier off — so the commit is the only variable and both numbers come from job logs
rather than from an entry's prose:

| | wall | mean TTFT | hits | published | evictions | pool peak |
|---|---:|---:|---:|---:|---:|---:|
| pre-fix 169d7bd | **198.32 s** | 5.14 s | **24/36** | 876 | 835 | 1999 |
| post-fix a43a379 | **403.01 s** | 10.83 s | **35/36** | 144 | 103 | 3735 |

**2.03x.** The pre-fix arm reproduces the 09-07 entry's 199.35 s to 0.5%, so that number was always
sound. Provenance: the pod tree was stamped `169d7bd` before the run and read back from the job log;
`compiles: clean` in both arms.

**Why every counter here except one is unreadable.** The entry population is what the commit changed —
876 published against 144 — so `evictions 835 vs 103`, `blocks_freed`, and every per-eviction yield
carry the commit in numerator *and* denominator. Reaching for `evictions 835 vs 103` as the mechanism
would have been the fifth withdrawn reading of this session.

**The exception is `hits`, whose denominator is turns, and turns are 36 in both arms.** Post-fix hits
**more often** (97% vs 67%) and is **twice as slow**. A fixed denominator makes that comparison sound,
and it forces one conclusion: each hit serves far less. At this pool and budget the deep entry is often
gone by the next turn, so the surviving match is the shallow one — and post-fix there is nothing
between 512 tokens and the full prefix, where pre-fix's 61 nested entries meant a row that lost its
deepest still had a near-deep one.

So #271 raised the hit *rate* and collapsed the hit *depth*, and the net is 2x. Its accept grid
counted token reuse at an unpressured capacity, where the deep entry never goes missing and the
intermediate entries look redundant.

**Fix direction: a bounded ladder** — a few spread publishes per row, keeping intermediate fallbacks
without returning to 62. Publishing only the last boundary is the **worst** available option, since it
removes the remaining fallback entirely. No revert: the flood was a real defect with a measured
cascade.

### The regression is entirely prefill, and the post-fix hit costs most of a miss

`mean_ttft × 36` is 185.0 s of pre's 198.32 (93.3%) and 389.9 s of post's 403.01 (96.7%), so the
deltas are **204.7 s wall against 204.8 s TTFT** — 0.15 s apart. The regression is time-to-first-token
in full. Nothing in decode, sampling or the tier contributes measurably, which retires every
mechanism that would have shown up as slower generation.

Split each arm's TTFT into a hit bucket and a miss bucket. Pre has 12 misses, post has 1, turns are 36
in both:

```
185.04 = 24·h_pre  + 12·m
389.88 = 35·h_post +  1·m
```

Two equations, one free parameter — the mean miss TTFT `m`. Solve at the 09-07 entry's ~14.1 s full
prefill: **h_pre = 0.66 s, h_post = 10.74 s**, against a miss of 14.1 s. The post-fix hit does **76% of
a miss's work.**

`m` barely matters, and that is the point:

| assumed miss cost `m` | h_pre | h_post |
|---:|---:|---:|
| 5.0 s | 5.21 s | 11.00 s |
| 10.0 s | 2.71 s | 10.85 s |
| 14.1 s | 0.66 s | 10.74 s |
| 15.42 s (ceiling: h_pre → 0) | 0.00 s | 10.70 s |

`h_post` is **10.7–11.0 s across the whole admissible range**, because the post arm has one miss out of
36 and `m` therefore carries 1/35 of the weight. So `h_post` is effectively measured, not inherited from
another arm's number — the concern about borrowing 14.1 s applies to `h_pre` and to the ratio, not to
the post-fix per-hit cost. And `h_post > h_pre` holds for **every** `m > 0`: the inequality reduces to
`2880.7 > -396·m`.

**A post-fix hit is a miss wearing a hit's label.** It matches, it reports a hit, and it re-prefills
nearly everything — which is exactly why the hit *rate* rose while the wall clock doubled. Post-fix
almost always finds the 512-token entry, so it almost never records a miss. Pre-fix's 12 real misses
were cheaper in aggregate than post-fix's 35 nominal hits.

That also corrects the mechanism as first stated. The claim was "the deep entry is never published"; it
is published, and found — it is published and then **evicted**, and post-fix the only thing left below
it is 512 tokens. Pre-fix's 61-entry chain meant the fallback after losing the deepest was still
near-deep. The defect is neither the publish depth nor the eviction: it is that **first+last leaves no
rung between 512 and the full prefix**, and every rung matters precisely because eviction is guaranteed
at this shape. What #271 removed is graceful degradation.

**Limits.** `h` and `m` are bucket averages over each arm's own turn population, and the populations
differ — prompts grow (`--grow 10`), so pre's 12 miss turns are not post's 1. The split is a two-bucket
model over measured totals, not an identity. And the depth claim is still inferred from time rather
than measured: `bench_chat_interleaved.py` does not record matched tokens per hit. That is one field,
it needs no card window, and it converts this from a model to a measurement — worth adding before the
ladder is written.

The earlier K-spaced refutation was measured in token-counted reuse on the CPU grid, which is the unit
that cannot see any of this. It should be re-tested in TTFT.

## What remains open

199.35 s (pre-fix) against 403.01 s (post-fix) **at the same 8192-block pool** is still a real
cross-commit difference. The pool finding explains cell357 versus the other cell, not pre- versus
post-fix at 8192. What is gone is any ability to attribute it to a publisher mechanism, since both
arms sit at 99.1% occupancy where block reclaim dominates. The decisive next arm is cell357's workload
with `--blocks` unset; only if 403 s survives that does the publisher return to scope.

Consequently **no publisher fix is justified**: last-only, K-spaced and the chunk-size knob were each
proposed to fix evictions the pool size explains.

## Rule

**Matching a cell's parameters exactly is only a control if you know which parameter is doing the
work.** `--blocks 8192` was copied for fidelity and was itself the independent variable. Fidelity
reproduces a confound as faithfully as it reproduces a condition.

And the through-line of all four refutations: **a measurement's unit, traffic shape, pressure presence,
pressure kind, and pressure axis are all operands, and the number carries none of them.** In one
session the substitutions were a token count for a prefill second, a partial sharer for a
continuation, an unpressured store for a pressured one, a count-pressured store for a byte-pressured
one, and a refcounted block union read as additive. Every one produced a clean, self-consistent table.
Two arms agreeing tells you nothing when both were measured against the wrong operand.

Operationally: before a number becomes a cause, name the operand it was measured against and check
that operand matches the failing configuration — not the configuration you meant to reproduce.
