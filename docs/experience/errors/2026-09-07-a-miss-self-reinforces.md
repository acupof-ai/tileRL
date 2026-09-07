# A miss self-reinforces: one prefill publishes 65 entries and evicts 64, into a budget of 6 — H20, 2026-09-07

**Status:** open — the mechanism is attributed, the fix is not written. Listed in OPEN.md.
**Date:** 2026-09-07
**Card:** H20 card 0, `qwen38-27b` NVFP4, tree `/work/tilerl-s-tierbench` sha `71db697`
**Run:** `tbalt`, 12 sessions × 3 turns, `--state-bytes 1073741824` (6 snapshots), tier off

## Context

The V100 tier cell read turn-0 hits on every *other* conversation — 3364-token
prompts hitting at 5.6 s and 3344-token ones missing at 19.1 s, alternating over
12 sessions. Three candidate causes were refuted before any card time:

- **Fixture contamination.** Measured: all 12 turn-0 prompts are distinct, diverging
  8 characters past the shared system prefix at the session index `_fillers`
  prepends. `prompts[0] == prompts[2]` is False. (This was the first diagnosis and
  it was wrong; retracted.)
- **Block alignment.** Refuted by the data already in hand: 3374 (mod 16 = 14)
  hits and 3354 (mod 16 = 10) misses, so alignment does not sort them.
- **An LRU rhythm.** `scripts/probe_session_parity_lru.py` runs the real
  `PrefixStore` over the bench's interleave — 16 arms, budget 4/6/11/13 × 2 or 6
  chunks × length-varying-by-parity or fixed. Every arm is bimodal, all-hit or
  head-evicted-after-session-0, and **never alternating**; the length-varying and
  fixed-length arms are identical row for row.
- **Boundary position shifting with length.** The engine's publish boundaries
  compute to 2560 and 3072 at all four observed lengths — identical.

So the alternation is V100-specific and stays open. What the H20 cell run to check
it produced instead is this entry: a mechanism the alternation had been hiding.

## The per-row instrument, and what it showed

`/health` publishes `prefix_entries` as a bare count and the bench read it only at
the end. Reading it per row (71db697) cost nothing — 199.63 s against the same
cell's earlier 199.35 s, **0.14%**, with `compiles: clean` on both — and the
earlier run's figure is now confirmed by an independent repeat.

The resident count is `5/6` on **all 36 rows**, one distinct value for the whole
run. Yet turn 2 loses 11 of its 12 hits. A count that never moves and a hit rate
that collapses are the same store, so eviction is happening *within* a request and
the count is restored before the next read.

Per-row publishes and evictions attribute it:

| turn | hits | ttft | published | evicted | entries |
|---|---|---|---:|---:|---|
| 0 | 11/12 | 0.41–14.23 s | 3 | 2 | 5/6 |
| 1 | 12/12 | 0.86–1.02 s | 5 | 4 | 5/6 |
| 2 | **1/12** | 1.41–14.24 s | **65** | **64** | 5/6 |

Turn 2's misses publish 65 entries and evict 64, each. Turn 0 and 1's hits publish
3 and 5.

## The mechanism

A hit prefills only the tail. A miss prefills from token 0 — and **the prefill
publishes at every chunk end** (`engine.py:1031`, `_finish_prefills`'s
`prefill_from % BLOCK_TOKENS == 0` branch), so a long miss floods the store with its
own partial prefixes and evicts, among them, the shared head it needed.

Computed over `engine.py`'s chunking (a chunk pads to a 64-multiple, the budget
caps at 512) for turn 2's 31,767-token prompt:

- **from 0** — what a miss does: 63 chunks, **62** of which publish
- **from 30,766** — what a hit would do: 2 chunks, **1** of which publishes

The last chunk does not publish here: `prefill_from >= len(pf.tokens)` takes the
`done` branch above, so the publishes come from the 62 *interior* boundaries — and all
62 satisfy `% BLOCK_TOKENS == 0`, checked over the chunking rather than assumed.
Against **65** published: 62 interior chunk publishes, the final publish at
completion, and the decode entries.

So the cost of one miss is not one re-prefill. It is 62 publishes into a
6-snapshot budget, which evicts every other session's head, so the next session
misses too and does the same — 11 of 12 in sequence, each paying 14.1 s. The single
survivor is conv A, which ran first while the head was still resident.

This is [a prompt's own publishes evict the prefix it
shares](2026-09-07-a-prompts-own-publishes-evict-its-shared-prefix.md) at
12-session scale. That entry is marked fixed, and its fix is real: it replaced the
*decode* publish site so one conversation holds `prefill_chunks + 1` entries rather
than one per decode boundary. The site left standing is the one this cell hits —
`prefill_chunks` itself is 62 publishes at a 31k prompt, which already exceeds any
budget a pressured card has, before a single decode entry is published.

## Why the tier hides it rather than fixing it

The DRAM tier arm of the same grid reads 12/12 at turn 2 with **0** evictions, and
55.82 s against 199.35. It does not stop the 62 publishes — it gives them somewhere
to go, so the head is demoted instead of dropped and promoted back on the next
lookup. The measured 3.57x is real and this is the mechanism under it. A card with
no host tier (the V100, `kv_cache.py:390-396`) has no such relief.

## Rule

**A miss is not one re-prefill; at a chunked publish site it is one publish per
chunk, and those evict the thing that would have made the next request a hit.** The
cascade is self-reinforcing and its cost scales with prompt length, not with the
number of sessions: 62 publishes at 31k tokens against a 6-entry budget turns one
cold session into eleven.

**A resident-count reading cannot see an eviction that a later publish restores.**
`prefix_entries` was `5/6` on every one of 36 rows across a run whose hit rate went
12/12 → 1/12. The counter that saw it was `prefix_evictions`, which is cumulative
per row; the gauge was constant by construction because each request ends having
refilled what it emptied. Read a rate, not a level, when the level is restored by
the same code that disturbs it.

## Results

| date | commit | machine | model | run | wall clock | turn-2 hits |
|---|---|---|---|---|---:|---|
| 2026-09-07 | 169d7bd | H20 card 0 | qwen38-27b NVFP4 | 12 sess, 1.0 GiB, off | 199.35 s | 1/12 |
| 2026-09-07 | 71db697 | H20 card 0 | qwen38-27b NVFP4 | same, per-row `ent` | 199.63 s | 1/12 |
| 2026-09-07 | 169d7bd | H20 card 0 | qwen38-27b NVFP4 | same + dram tier | 55.82 s | 12/12 |

Raw artifacts: `/work/tbalt.log` on the H20 (36 rows, per-session totals,
`final_stats`), `/work/tb1w.log`, `/work/tb2w.log`.

## Open

- The V100 alternation itself. It did not reproduce here, so it needs the V100 grid
  with the per-row instrument to either show up or not. Two refutations above stand
  either way.
- The fix for the prefill publish site. The decode site's REPLACE pattern does not
  transfer directly: a prefill chunk's entry is what a *later* request matches
  against, so retiring the previous chunk would cost the intermediate prefixes that
  make a partial hit possible. Sizing that trade needs its own measurement.
