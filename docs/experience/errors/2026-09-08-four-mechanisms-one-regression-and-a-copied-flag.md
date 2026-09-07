# Seven withdrawn readings for one regression, and the cause was entries-per-row against the budget — H20 + CPU, 2026-09-08

**Date:** 2026-09-08
**Machine:** H20 pod card 0 (cells), local CPU target (diagnosis)
**Status:** open — the regression is measured: **1.81–2.12x on 16 lines of `engine.py`**, 100% TTFT, and
a hit costs the full prefill times the fraction it did not match (**median 0.9% error over 35 turns, one
parameter, nothing fitted**). Depth is a **512×k ratchet in submission order**, so first+last leaves each
session a match set by its queue position rather than a constant. The forward fix has not landed and may
not fit this budget — see the sizing criterion. Listed in [OPEN.md](../OPEN.md) with the latent
`_demote_one` count guard.

> Sections are in the order they were written, so **nine withdrawn readings** stand as the record.
> **The settled result starts at
> [Resolved](#resolved-188203x-across-16-lines-and-the-mechanism-is-one-count-vs-budget-tradeoff)
> and is pinned by
> [the measured model](#measured-a-hit-costs-the-full-prefill-times-what-it-did-not-match).**
> Withdrawn, in order: four mechanisms for the discrepancy, a solved-for TTFT split, a
> population-skew objection, a single-variable claim against a 7-commit range, a cold-prefill doubling
> that was one noisy row, and — on the measured data itself — a falsification of the mechanism that was
> an operand error. Every one was self-consistent when written.
>
> The first eight fell to arguments. The ninth is the only one where something **true** was withdrawn,
> and the reason the model survives where seven mechanisms did not is that it is a fit rather than an
> argument: it predicts out of sample, so it cannot be talked into or out of.

## Context

[#271](../wins/2026-09-08-cut-the-prefill-publish-flood.md) cut prefill publishing from every interior
chunk boundary (62 entries per 31k prompt) to the first interior boundary plus the last (2 per row).
Accepted on a CPU token-reuse grid: 104448 vs pure LRU's 81408, +28%.

Re-running [the 09-07 DRAM tier cell](../wins/2026-09-07-the-dram-tier-is-357x-when-the-budget-is-pressured.md)
at its exact parameters to settle that entry's Pending flag produced 403.01 s against its 199.35 s,
and the tier that had been worth 3.57x there promoted nothing. Four mechanisms were proposed for that
over the next two hours. All four were refuted, each because it was confirmed against a different
operand than the one the card ran. Three more readings fell after them.

The regression is real and the cause is #271. What took seven withdrawals was saying **why**, and the
answer is not the one this entry spent its first half looking for: not the tier, not the pool, not the
publisher's block retention, but the number of published entries per row measured against how many the
budget can hold.

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

## Resolved: 1.88–2.03x across 16 lines, and the mechanism is one count-vs-budget tradeoff

The single-variable pair is **`45acd87` → `a43a379`**, which are parent and child
(`git rev-parse a43a379^` = `45acd87`): 16 lines of `engine.py` — the `interior_published`
counter and the `== 1 or last` gate — plus a `ent=` field in the bench and docs. Every flag,
the workload, and the prompt tokens row-for-row are identical.

| | wall | hit n | mean hit TTFT | miss n | mean miss TTFT |
|---|---:|---:|---:|---:|---:|
| parent `45acd87` | **198.47 s** | 24 | **0.715 s** | 12 | 14.031 s |
| child `a43a379` run 1 | **403.01 s** | 35 | **10.342 s** | 1 | 28.050 s |
| child `a43a379` run 2 | **373.14 s** | 35 | **9.891 s** | 1 | 13.940 s |
| child + counter run 3 | **372.92 s** | 35 | **9.880 s** | 1 | 13.960 s |

**1.88–2.03x**, and the hit costs **13.8–14.5x** the parent's. That range prices only the child's
variance: the parent is a **single draw**, so its own bar is unmeasured, and if it shares the child's
~8% (three runs: 403.01 / 373.14 / 372.92 s) the honest envelope is **1.81–2.12x**. The two claims do not need the same confidence — a 4.6%
spread on the child's hit mean against a 14x effect threatens neither its sign nor its size, while an
8% spread on a wall clock is most of the gap between 1.88 and 2.03. The child hits *more often*
(35/36 against 24/36) and each hit is an order of magnitude dearer. The regression is
**100% TTFT**: the three turn deltas sum to 204.47 s against a 204.54 s wall delta on run 1.

`169d7bd` also ran at these flags (198.32 s), matching the parent to 0.08% and reproducing the
09-07 entry's 199.35 s to 0.5% — so the pre-#271 number was always sound, and #272 in between
changes nothing measurable (its cold miss is 14.07 s against 169d7bd's 14.02 s).

### The mechanism: entries per row against the budget decides whether depth binds or survival binds

One sentence covers all three turns, and the sign flips inside the run:

| turn | parent (61 rungs) | child (2 rungs) | delta |
|---|---|---|---:|
| 0 | deep hit, **0.4–0.6 s** | shallow hit, **11.0–17.7 s** | +143.99 s |
| 1 | deep hit, **0.8–1.0 s** | shallow hit, **10.5–16.2 s** | +114.76 s |
| 2 | 11/12 **MISS** at 14 s, `evict=64` | shallow hit, **6.8–12.2 s** | **−54.28 s** |

The parent's turn-2 misses are the flood evicting itself — 61 entries per row against a
6-snapshot budget, so a row's own turn-1 entries are gone by turn 2. That is exactly the defect
#271 was written to fix, and at turn 2 the child genuinely wins by 54 s. The child's 2 entries
survive. So:

- **parent**: deep rungs, self-evicting → fast early, misses late
- **child**: shallow only, survives → slow early, hits late

Net at this shape: **−144 −115 +54 = the child loses by 205 s**, because turns 0 and 1 lose more
than turn 2 gains. This is why #271's accept grid liked the change: the grid measured reuse at
admission, which both a self-evicting flood and a shallow survivor score well on, and it never
priced the early turns in seconds.

### The fix has a sizing criterion, and it may not be satisfiable at this budget

The two arms are the endpoints of a count axis and both endpoints lose something. K rungs
spanning the range is not a compromise — it is the only region where depth is available early
**and** the entries survive to turn 2. The constraint is
**K × snapshot_bytes ≤ budget**, so a row's own rungs never evict each other.

At this cell that is a hard limit rather than a free parameter: the budget holds
**6 snapshots for 12 sessions**, so K ≥ 2 per row already oversubscribes it. **The honest reading
is that no publisher wins both ends at this budget** — the ladder is the right shape for a card
with more budget per session, and this cell's answer is that its budget is too small. The earlier
K-spaced refutation measured token-counted reuse, the one unit that cannot see either failure;
it should be re-tested in TTFT at a budget where K ≥ 3 fits.

**The measured model prices K, so this stops being a judgement call — and it adds a second constraint
that bites harder than the byte one.** A hit costs `2.51 + 10.45 × (1 − depth)`, so a rung lifting a
row's match from *d* to *d′* saves `10.45 × (d′ − d)` — **0.10 s per percentage point of depth**. But the
2.51 s fixed term is paid per *hit*, not per rung, so a rung only earns its place if the depth it adds
saves more than the alternative of matching nothing extra:

**`10.45 × Δdepth > 2.51` → Δdepth > 24.0% of the prompt.**

**Rungs spaced closer than ~24% of the prompt are net negative**, which caps K at about 4 on depth
grounds before the byte budget is consulted — and the byte budget already says K ≈ 1 here. Both
constraints point the same way at this cell, from different data.

The ratchet says who needs the rungs: the early sessions in the queue, matching 1.7–8.5%. So the
arithmetic to run before writing any ladder is `10.45 × Δdepth × hits` against the snapshot bytes the
rung takes from the store, and it needs no card window.

**No revert.** The flood is a real defect with a measured cascade, and the child is genuinely
better at turn 2.

**Depth is measured**, not inferred — see the section below. A hit's cost is the full prefill times
the fraction it did not match, to a median of 0.9% over 35 turns.

### The buckets were per-turn columns in the log, and two of us modelled them instead

The table above is read off `ttft=` and `hits=` on each turn row. It was nearly reported as a
derivation instead: two of us wrote the two-bucket system

```
185.04 = 24·h_pre  + 12·m
389.88 = 35·h_post +  1·m
```

swept the one free parameter `m` (the mean miss TTFT), and concluded `h_post` sits at 10.7–11.0 s for
every admissible `m` because the post arm has 1 miss in 36. The conclusion was right and the range for
`h_pre` (0.66–7.70 s) was honest and useless. `m_pre` measured **14.03 s**, which is the 14.1 s the
model had imported from the 09-07 entry — so the model reproduced the number it was built from, and
that agreement read as validation. **A model that reproduces the authority it borrowed from is not
corroboration, and it feels exactly like corroboration.** Before modelling a quantity, grep the log
for it.

A second withdrawal came from the same rows. The population objection — prompts grow under
`--grow 10`, so misses skew early and short, so the true `m_pre` is below 14.1 s, so `h_pre` is higher
and the collapse softer — has the sign backwards. Misses are the **longer** prompts (31689 against
30528). The chain was valid and the premise about which turns miss was wrong; only the rows could say.

**One earlier reading survives in corrected form.** The first mechanism written here was "the deep
entry is never published", and it is published and found. What happens is that it is published and
then **evicted**, so post-fix the only thing left below it is 512 tokens — which is the
count-vs-budget tradeoff above, stated from the child's side only. The full statement needs both
sides, because the parent's flood evicts *itself* and that is what turn 2 shows.

### Measured: a hit costs the full prefill times what it did not match

`prefix_hit_tokens` ran on the child's publisher (branch HEAD = `a43a379` plus the counter,
`git diff a43a379 HEAD -- src/` is 5 lines). It converts the mechanism from inferred to measured, and
it corrects the shape of the claim.

**Depth is a ratchet, not a constant.** Cumulative `prefix_hit_tokens` after 35 hits is **322560**,
which is `512 × 35×36/2` exactly. The k-th hit in submission order matches **512 more tokens than the
k−1-th**: 512, 1024, 1536 … 11776, i.e. 1.7% → 56.3% of the prompt. The 512 this entry had been
quoting was **conv B, the first and smallest hit in the run**, read as the typical one.

**And TTFT is linear in the unmatched fraction.** Regressed over all 35 hit turns
(`err = (pred − obs)/obs` throughout):

| model | R² | residual sd | shallow (<15% depth) | deep (>40%) |
|---|---:|---:|---:|---:|
| `miss × (1 − depth)`, no fit | 0.8824 | 6.0% | **+6.6%** | **−8.7%** |
| `2.51 + 10.45 × (1 − depth)`, OLS | **0.9947** | **1.4%** | +0.6% | +0.4% |

The one-parameter form is good on average (median −0.9%) and its residuals **march monotonically with
depth**, +6.6% shallow to −8.7% deep — a 15-point trend, which is structure absorbed into an average,
not noise. Adding an intercept removes the trend entirely and takes the residual sd from 6.0% to 1.4%.

So the measured cost of a hit is **2.51 s fixed plus 10.45 s × the fraction it did not match**, and both
terms mean something:

- **The 2.51 s is depth-independent** — paid at any match length, and it would be paid at 100%.
- **The slope is 10.45 s, not the 13.96 s miss.** Unmatched tokens re-prefill *cheaper per token* than a
  cold miss, which is what a prefill starting with a populated block table and a warm pool should look
  like.

**Where the 2.51 s does NOT come from.** The obvious candidate is the state restore, since a snapshot is
a constant size at any prefix length (`kv_cache.py:1434`) and a depth-independent cost is exactly that
shape. **The arithmetic refuses it**: this checkpoint's snapshot is 156.9 MiB, which is 0.17 ms at HBM
bandwidth and 82 ms even at a slow 2 GB/s host copy — three to four orders of magnitude short of 2510 ms.
So the shape fits and the magnitude does not, and the term is **unattributed**. Naming the restore here
would be the tenth withdrawal.

This closes the honesty limit that mattered: a shallow hit and a deep hit that stalls on something else
are now distinguishable, because a stall would appear as depth-correlated residual and after the
intercept there is none (+0.6% shallow, +0.4% deep). What replaces it is a smaller, sharper open
question — what the 2.51 s is.

**What the ratchet says about the publisher.** 512×k across sessions in submission order is not a
property of any row's prompt — it is the store serving one more block-chunk to each successive session.
So under pressure, first+last does not give every row a 512-token match; it gives a row a match whose
depth is set by its **position in the queue**, with the earliest sessions worst off. That is a sharper
statement of the same defect and it is what the ladder has to fix.

**One withdrawal on this data, and it is the only one where something true was withdrawn.** The first
reading of these rows was that the mechanism was falsified: depth rises 22x while TTFT goes 12.80 →
10.93 s, which looks flat. That compared depth against **absolute** TTFT across turns whose prompts also
grow. Dividing by what was left to prefill gives the table above. Testing a claim about a ratio by
looking at its numerator — the same operand error this entry's Rule section names, made on the entry's
own instrument within the hour.

### Not one variable at first, and the cold miss was noise: two withdrawals

**"The commit is the only variable" was written against `169d7bd..a43a379`, which is 7 commits.**
That went into an entry and a PR body before anyone counted the range. The claim is true for the
parent pair above and was false as first stated; the fix was to find the right bracket, not to
abandon the claim. `git log --oneline A..B | wc -l` is the check and it costs one command.

**And a "cold-prefill doubling" was reported from a single row.** Turn 0 conv A read 28.05 s against
the parent's 14.07 s — a full prefill on an empty store, where no publish policy can reach — so it
looked like a second, unattributed effect. Three things killed it:

1. **It is 6.8% of the regression.** Per-turn, the delta is +143.99 / +114.76 / −54.28 s; that one row
   contributes +13.98 s and everything else +190.49 s.
2. **The neighbouring rows degrade the same way.** Conv B..L go from 0.41–0.58 s to 10.99–17.74 s at
   *identical* prompt tokens, and every one is a hit in both arms. Conv A is simply the only row with
   no hit to degrade, so its degradation appears as a doubled miss instead of a shallow hit.
3. **It does not reproduce.** A second `a43a379` run at the same flags reads **13.94 s** on that row,
   against the first run's 28.05 s.

Two arguments were built on that row before it was re-run, and both were over-fitting one sample: that
its 2.0007x exactness was a structural signature (thermals do not land on 2.000), and that the
publisher change predicted the opposite sign so a mechanism was missing. Neither survives a row that
moves 14 s between runs. **The point-vs-mean error was made three times inside one investigation** —
comparing one row to a 12-row mean, positing a 12-row distribution to explain that gap, then reading
exactness off the same row — and the 12 rows were in the file every time.

**What the re-run also exposed: the headline had unmeasured spread.** `a43a379` is 403.01 s and
373.14 s on two runs, 8.0% apart, so the ratio is **1.88–2.03x** rather than 2.03x — and the parent is
still a single draw, so even that range is half an envelope. Every earlier version of this entry quoted
one run as the result.

The cleanest statement of the error is that **the cold row's same-commit spread is 28.05 / 13.94 =
2.01x**, numerically identical to the cross-commit "effect" a structural argument was built on. A
signature and a coin flip were the same number, quoted to four digits.

### The `compiles: clean` on every cell of this grid was vacuous

All four cells report `compiles: clean`. All four serve logs are **0 bytes**.

`_compiles` counted marker lines and returned that count, so an empty-but-existing file returned 0,
`known = all(compiles >= 0)` held, `dirty` was empty, and the verdict printed `clean`. The logs are
empty by construction: the arms run the server as `python3 -c ... > /work/<name>-serve.log` with no
`-u`, so stdout is block-buffered to a file, and each arm ends with `kill $SRV` — SIGTERM, no flush.

Measured on the pod, same script, same SIGTERM, `-u` as the only variable:

| | bytes after 3 s | bytes after SIGTERM | marker found |
|---|---:|---:|---:|
| `python3` | 0 | 0 | no |
| `python3 -u` | 38 | 38 | yes |

`TILELANG_PRINT_ON_COMPILATION` defaults to `"1"` (tilelang `env.py:371`), so the marker *is* emitted
on every compile — the empty log is the instrument, not a compile-free run. **A JIT inside a measured
turn would have been charged to the tier and read as clean**, which is precisely the confound
`--server-log` exists to exclude, and it makes the cold-miss doubling unresolvable from these logs.

Fixed: `_compiles` returns -1 for an empty file, so the verdict reads
`unknown (no --server-log, or it is empty -- run serve under python3 -u)`. The distinction that has to
survive is that a log **with** content and no marker is still genuinely clean, or `clean` becomes
unreachable and the gate is useless in the other direction — `tests/test_bench_compiles_verdict.py`
holds both directions and was verified red against the original.

**The depth run is the first time this verdict has meant anything on this grid**: `compiles: clean`
against a **164-byte** log, flushed because serve ran under `python3 -u`. Before the fix the same word
came from a 0-byte file, so the fix's value would otherwise be invisible in the output.

**And the vacuity is narrower than it first looked, which took two controls to establish.** The 09-07
cells that licensed 3.57x also report `compiles: clean`, and their logs are **164 bytes** — flushed, 0
markers. That is a different thing:

- **The marker reaches stdout.** tilelang installs a `TqdmLoggingHandler` on import and raises the
  `tilelang` logger to INFO (effective level 30 → 20, `propagate` False), so `logger.info` is not
  dropped by the root logger's WARNING default. Measured: the marker lands on **stdout**, and every arm
  redirected `> log 2>&1`. A missing marker is not a stream mismatch.
- **A cold cache does emit it in the same redirect shape.** Same script, same `python3 -u`, same
  `> log 2>&1`, `TILELANG_CACHE_DIR` as the only variable: cold gives **1 marker / 270 bytes**, warm
  gives **0 markers / 24 bytes**.

So on the 09-07 cells `compiles: clean` is a real reading — warm cache, zero JIT — and that entry's
verdict stands. The vacuity is specific to the 09-08 grid, whose logs never flushed. The 0-byte case and
the warm-cache case print the same word and mean opposite things, which is why the fix separates them
instead of treating every 0 as suspect.


## What remains open

Superseded, and kept because the reasoning is the fifth refutation. This section read: "no publisher
fix is justified — last-only, K-spaced and the chunk-size knob were each proposed to fix evictions the
pool size explains", and the decisive next arm was named as cell357's workload with `--blocks` unset.
Both were wrong.

The `--blocks`-unset arm ran (2560 blocks, not the 48099 intended, because `--max-ctx 40960 --slots 16`
still bound it): **424.20 s at 78.1% peak against cell357off's 403.01 s at 45.6%**, evictions 104 vs 103.
Tripling the pool changes nothing, so the pool is a threshold and not the cause. And the cross-commit
arm above then put the publisher squarely back in scope. What survives is the narrower claim the
`pub271off` arm supports on its own: 139 evictions at **8.8%** occupancy cannot be capacity.

Still open:
- **What the 2.51 s per-hit fixed cost is.** Measured at R² 0.9947 and unattributed. The state restore
  has exactly the right shape — constant at any prefix length — and is 0.17 ms at HBM bandwidth against
  2510 ms observed, so it is not that. This is the sharpest open question in the entry and it needs a
  phase-attributed profile of one hit, not an argument.
- **The ladder, if any budget admits one.** Two independent constraints now, and they agree:
  `K × snapshot_bytes ≤ budget` gives K ≈ 1 here, and `10.45 × Δdepth > 2.51` caps useful rung spacing
  at ~24% of the prompt. The arm is the same pair at a budget where K ≥ 2 fits, measured in TTFT.
- **The per-eviction yield split** — per eviction, the count of the entry's blocks where
  `refcount[b] > n`, logged with the entry's token length.

Closed by the depth run: whether the mechanism is hit depth (yes, R² 0.9947), and whether `compiles`
can now go red (yes — 164 bytes, flushed).


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

Five more, each from a mistake made *after* the four above were written up:

**Before modelling a quantity, grep the log for it.** Two sessions solved a two-equation system for the
hit and miss TTFT buckets, swept the free parameter, and argued about the bound. Both columns were
printed on every turn row of both arms. The model agreed with the log because the log is where its one
constant came from, and a model that reproduces the authority reads as confirmation.

**"One variable" is a claim about a commit range, so count the range.** `169d7bd..a43a379` is 7
commits. Naming the two endpoints and matching every flag between them makes the *configuration*
single-variable and says nothing about the code. `git log --oneline A..B | wc -l` is the check, and it
costs one command.

**A green verdict needs a file that could have made it red.** `compiles: clean` on four cells, four
0-byte logs: `_compiles` returned 0 for an empty file, and the server ran without `-u` so SIGTERM
flushed nothing. The negative control is not "does the parser count markers" — it is "can this file
ever contain one". Check the instrument's input is non-empty before reading its output as a result.

**A single row is not a distribution, and this was got wrong three times in one investigation.** The
28.05 s cold miss was compared against a 12-row mean (mine); a 12-row distribution was then posited to
explain the gap (a peer's); then its 2.0007x exactness was read as a structural signature, since
thermals do not land on 2.000 (mine again). The 12 rows were in the file for all three. A second run
put that row at 13.94 s. Exactness in one sample is not evidence of mechanism — a single draw has no
shape to be a signature of.

**One run is not a measurement of a ratio.** `a43a379` came back 403.01 s and 373.14 s at identical
flags, 8.0% apart, so the honest figure is 1.88–2.03x. Every earlier version of this entry quoted the
first run as the result. A ratio built from two single runs hides two error bars, and the one that
mattered here was large enough to move the headline.


