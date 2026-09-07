# The eviction policy was the wrong layer: six policies, and the two best numbers were bugs — CPU, 2026-09-08

**Date:** 2026-09-08
**Machine:** local CPU target, tiny model
**Status:** closed — the defect is fixed at the publisher instead
([wins/2026-09-08-cut-the-prefill-publish-flood.md](../wins/2026-09-08-cut-the-prefill-publish-flood.md))

## Context

[A miss self-reinforces](2026-09-07-a-miss-self-reinforces.md) attributed the H20's 199.35 s cell:
one 31k-token miss publishes 62 prefix entries into a budget holding 6, evicting the head every
other session shares. Two readings of that were available — *LRU picks the wrong victim*, or *the
publisher emits too many entries*. This entry is the record of taking the first one, for six
policies, and being wrong.

## The six policies

`scripts/probe_prefix_eviction_policy.py`: 6 sessions × 2048 tokens × 2 turns through the engine,
reuse in tokens read at admission, 196608 possible.

| policy | grid | cells worse than LRU |
|---|---:|---:|
| pure LRU (baseline) | 81408 | 0 |
| extensions-before-heads, LRU among them, **+ reparent** | 83968 | 3 |
| two-class (recent sharer evicted last), window 4 | 83968 | 1 |
| extensions-before-heads, longest first | 101376 | 2 |
| capped sharers (`min(sharers, 2)`) | 109568 | 4 |
| middles-before-leaves | 110592 | 4 |
| length × sharers | **115712** | 1 |
| extensions-before-heads, LRU among them | **130560** | 3 |

The two best are both artefacts.

## 130560 was an orphaned pointer

The policy protected an entry when `entry.parent is not None and entry.parent in self._by_id`.
Dropping a **middle** entry left its children pointing at a dead eid, `parent in self._by_id` read
False, and those orphans read as family roots — so the policy protected them. That accidental
protection was the whole gain. Closing the chain in `_drop` took 130560 to **83968**, cell for cell:
`(0,12)` 9216→3072, `(512,8)` 9216→3072, `(1024,8)` 11776→6144.

Found by a gate at a 2-entry budget over **two** families: `resident [(2, 48, 1), (8, 16, None)]`
with family 1's true head gone. The longest-first variant never hit it because it always evicts a
leaf.

What the accident was doing pointed at the next policy: it protected the **deepest** prefixes, not
the heads, and a request matches the longest stored prefix, so the deepest entry is the only one
that saves its row tokens.

## 115712 was LFU pollution

`length × sharers`, where `sharers` rises on a lookup match and on a dedup re-publish. It never
falls. A prefix that was popular an hour ago keeps its weight forever.

Sweeping the number of stale formerly-hot prefixes (each matched 10 times, then never again)
against a live workload — reuse in tokens, pure LRU is flat 1152:

| stale hot prefixes | length × sharers | LRU |
|---:|---:|---:|
| 1 | 1024 (−6%) | 1152 |
| 2 | 960 (−12%) | 1152 |
| 4 | 832 (−24%) | 1152 |
| 8 | **64 (−94%)** | 1152 |
| 16 | 64 (−94%) | 1152 |

At a budget of 8 with 8 stale entries the resident set is 8/8 stale: the head and every session's
own entry are gone. Arithmetic — a stale 64-token prefix at 10 sharers weighs 704, against a
session's own 128-token entry at 2 (256) and the shared head at ~7 (448).

**The accept sweep could not see this: every prefix in the 4×4 grid is live.** The grid tests a
weight's ranking, not its memory. A monotonic counter needs an aging arm and the grid has none.

## No decay rate fixes it

Two clocks — increment per match, decay per eviction, with `seen` recording the eviction count at
the last match — was the obvious repair. Fixture P is the pollution sweep (LRU = 1152, perfect);
fixture H is a live re-matched head against one-shot bursts (head served / reads).

| policy | P: 6/8 | P: 8/8 | P: 8/16 | P: 12/16 | H: 6 every 2 |
|---|---:|---:|---:|---:|---|
| LRU | 1152 | 1152 | 1152 | 1152 | 1/4 |
| length × sharers | 64 | 64 | 64 | 64 | 4/4 |
| aged, half-life 2 | 704 | 448 | 704 | 704 | 1/4 |
| aged, half-life 4 | 192 | 64 | 704 | 448 | 2/4 |
| aged, half-life 8 | 64 | 64 | 64 | 64 | 4/4 |
| capped at 2 | 1152 | 1152 | 1152 | 1152 | 2/4 |
| two-class, window 4 | 1024 | 896 | 1088 | 1088 | 2/4 |

The rate that clears pollution drops the head; the rate that keeps the head pollutes. Monotone
across four values, so **no single rate satisfies both**. Capping clears pollution perfectly and is
the worst policy on H — it throws the signal away. Two-class holds both, and its grid is 83968.

## A framing error the measurement corrected

The head fixture was first written re-reading the head every burst, and **plain LRU passes that
8/8**: `lookup` calls `move_to_end`, so a re-matched entry is refreshed. Recency already protects a
hit — for exactly as long as it takes the publishing row to emit its next chunk. Only the column
where the burst between re-reads outpublishes the budget discriminates any of these policies. A
fixture that does not is measuring `move_to_end`.

## Two vacuous instruments

- **Reading `lookup` after the request finished.** By then the row's own publishes are in the
  store, so both arms score a hit the row just created for itself. Under that reading pure LRU
  reused 2064 tokens/session against the fix's 512 — the fix looked refuted and the number was the
  instrument's. Reading the **first** lookup of each request separates them.
- **A per-request counter keyed on `id(pf)`.** The monkeypatched publish-arm harness kept
  `{id(request): boundaries_seen}`; CPython recycles ids, so a fresh row inherited a completed
  row's count and skipped its own first boundary. It reported **+38% / −23%** where the real tree
  edit measures **+28% / −17%** — pessimistic on the gain and optimistic on the cost, the worst
  combination, and it would have shipped as a claim. The real fix keeps the counter on `_Req`,
  which cannot alias.

## Also refuted, before these

- **REPLACE at the prefill publish site**, mirroring the decode site's fix: 5 hits → **0**. A
  prefill chunk's entry is what a *later* request matches against, so retiring the previous chunk
  destroys cross-session sharing.
- **Hit-aware eviction**: a no-op. At eviction time every entry has `hits == 0`, by construction —
  the publishes causing the pressure have not been matched yet.

## Rule

**When a cache defect's operand is a count, no victim choice fixes it.** Six policies against a
publisher emitting 62 entries into a budget of 6; the best honest one gained 3%. Cutting the count
to 2 gained 28% with plain LRU untouched.

**A monotonic weight is a bug until an aging arm says otherwise.** Both the grid and the head
fixture are made of live entries, and both passed the policy that collapses 94% under stale ones.

**Two numbers that look like wins deserve the same scrutiny as a number that looks wrong.** 130560
and 115712 were the two best results in the table and both were defects. The reflex to check
arrived only when a third policy contradicted them.
