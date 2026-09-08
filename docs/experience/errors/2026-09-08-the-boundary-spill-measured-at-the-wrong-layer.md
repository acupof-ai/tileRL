# The boundary spill was measured at the wrong layer, and the interface that survives it — 2026-09-08

> Status: **the `src/` change was written, measured, and NOT landed.** `PrefixStore.spill_held`
> exists as a reviewed design with its tests and mutation checks (PR #287, split so only this
> entry lands). It is not in the tree. The reason is not that the code was wrong — it is that
> the number it was built for turned out to belong to a different layer, and the tier it
> writes to is under a REJECT that is **narrow and expiring**: it covers this tier with its
> unreachable load path, not the idea, and explicitly does not transfer to the block-granular
> store. When either changes, this comes back.
>
> Two figures must never be quoted as one "yield": **`spill_yield`** (boundary entries that
> can reach the disk tier) and **`match_yield`** (boundary entries a later `lookup` can still
> hit). The DRAM tier moves only the second, from 0% to 100%. A single number here is a
> conflation, not a measurement.

## Context

[The boundary entry](../errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md) closes
`budget == 512` and stays open for the rest: 2448 of 35756 prompts in the default 504–512
window spill nothing, because `_last_prefill_boundary` takes `n` where the boundary is
`f(n, budget)`. Four fixes were measured and rejected. The one that survived — capture the
snapshot at the boundary, spill it later — needs the store to accept a spill for an entry it
already holds, which it does not: `insert`'s duplicate check returns before its write-through,
so re-offering held tokens with `spill=True` leaves the tier's `offered` at **0**. Not a
refusal, no counter moves, indistinguishable from a spill that happened.

That gap is real and the store-side fix for it is straightforward. The mistake was not in the
fix. It was in never asking whether the entry is still there to spill.

## What the measurement found

A boundary entry has to survive from its publish to DONE, and in that window every other row
in the batch publishes its own prompt-complete entry. Driving the real `PrefixStore` with
every row publishing its boundary first and reaching DONE afterwards:

| batch | snapshot slots | spillable | gone | `spill_yield` |
|---:|---:|---:|---:|---:|
| 4 | 9 | 4 | 0 | 100.0% |
| 8 | 9 | 1 | 7 | **12.5%** |
| 16 | 9 | 0 | 16 | 0.0% |
| 8 | 16 | 8 | 0 | 100.0% |
| 16 | 17 | 1 | 15 | 6.2% |
| 16 | 32 | 16 | 0 | 100.0% |

The first reading of this was a threshold — `2 × batch ≤ slots`, 100% one side and ~0% the
other. **It is a ramp, and the threshold was a sampling artefact**: the four slot counts first
sampled were the two ends and one midpoint. One slot at a time at batch 8:

```
slots  6  7  8  9 10 11 12 13 14 15 16 17
spill  0  0  0  1  2  3  4  5  6  7  8  8
```

```
spillable = clamp(snapshot_slots - batch, 0, batch)
```

Zero mismatches over batch ∈ {2, 4, 8, 16, 32} against slots 0..2b+2, reproduced independently
by two sessions with different constants. It is also **forced rather than fitted**: the `b`
boundary snapshots publish first, the `b` prompt-complete ones are all newer, and the `s`
survivors are the `s` newest — so boundary survivors are `s − b`, capped at `b`. Derivable
without running anything, which is why the two reproductions agree exactly.

The default lands low: `state_bytes` is `mem_get_info()[0] // 4` (`engine.py:1676`), priced at
**9** resident snapshots on a V100 at the 27B's 144 MiB, against `max_batch = 8`
(`engine.py:195`). `clamp(9 − 8, 0, 8) / 8 = 12.5%`.

## The two yields, and why the tier does not rescue the spill

`_entries_capacity` (`kv_cache.py:1477`) puts the host budget in the same denominator:

```python
avail = self.state_bytes + (0 if self._dram is None else self._dram.budget_bytes)
return min(self.capacity, avail // self._snapshot_bytes)
```

So raising the tier budget raises the reported capacity. Holding HBM at 9 slots:

| batch | dram slots | `entries_capacity` | spillable | demoted | gone | `spill_yield` | `match_yield` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 9 | 1 | 0 | 7 | 12.5% | 12.5% |
| 8 | 8 | 17 | 1 | 7 | 0 | 12.5% | **100.0%** |
| 8 | 64 | 73 | 1 | 7 | 0 | 12.5% | **100.0%** |
| 16 | 0 | 9 | 0 | 0 | 16 | 0.0% | 0.0% |
| 16 | 64 | 73 | 0 | 16 | 0 | 0.0% | **100.0%** |

`spill_yield` is untouched by the tier at every size. `match_yield` goes to 100%, because
`lookup` promotes a demoted entry in place (`kv_cache.py:1253-1266`) — 12.7 ms of pinned H2D
against the 163 s re-prefill the comment there prices it against — so the entry stays
matchable.

**"The tier only renames the failure" was stated, adopted by a second session, and relayed to
ckl before it was withdrawn.** It is true of spill and false of match. `no-entry` is a
permanent loss; `demoted` is an entry that is intact, indexed, and one H2D from serving.

## Why the fix did not land

Not the 12.5%. Three things, and any one of them is enough:

1. **The spill goes to the SSD tier, which is under a recorded REJECT on the serve path** —
   1.65x worse wall clock per turn, 0 hits at 12 sessions
   ([errors/2026-09-06](../errors/2026-09-06-the-ssd-tier-is-165x-worse-at-12-sessions.md)) —
   and `--ssd-path` defaults empty (`cli.py:112`). So 12.5% of 2448 is ~300 prompts spilling
   into a tier that is off. **That reject carries two qualifiers of its own and citing it
   without them overstates this**: it says *this* tier, whose load path is unreachable, cannot
   repay its write cost — not that an SSD tier is a bad idea — and it explicitly does not
   transfer to the block-granular store `AGENTS.md` names as the upgrade, against which the
   verdict has to be re-run. So this reason expires when either the load path becomes
   reachable or that store lands, and `spill_held` comes back with it.
2. **For a second-turn hit, the DRAM tier already delivers 100%**, so the fix covers only the
   configuration with no tier at all.
3. **For cross-restart persistence the tier cannot help**, and the thing that could is the
   rejected one.

Both arms of "what is the boundary publish *for*" land in the same place.

The publisher-layer fix also addresses the wrong constraint. Candidate 3 fixes *the boundary
position is mispredicted*; the binding constraint is *the boundary entry does not live to
DONE*, which is capacity, same root as the sizing row in
[the four-mechanisms entry](../errors/2026-09-08-four-mechanisms-one-regression-and-a-copied-flag.md).

## The interface, which is the durable part

Kept because the capacity side is fixable and this comes back. Three conclusions, each with
the mechanism rather than a preference:

**A string return, not a bool.** Two failure modes are invisible to a caller that checks
whether the entry is present:

| pressure | entry | `state` | blocks |
|---|---|---|---|
| DRAM demote | present | **None**, `demoted=True` | refcount 1 |
| count pressure | **gone** | — | — |
| no pressure (control) | present | held | refcount 1 |

`spilled` / `no-tier` / `no-entry` / `demoted` / `no-state` / `resident` / `refused`. A bool
hands the engine half a silent no-op on exactly the configuration the tier exists for — the
same shape as the bug being fixed.

**Spill `entry.blocks`, never the caller's.** At DONE the request's block list has grown past
the boundary entry, so a caller-supplied list would describe a longer prefix than the entry
holds. The blocks are the one thing that cannot vanish: `insert` retains them, refcount 1 in
both pressure cases, measured.

**`demoted` must not promote-then-spill, and this is mechanical.** `lookup`'s promote adds the
bytes back and returns without re-entering `insert`'s
`while len > capacity or _state_used > state_bytes`:

```
after 5 publishes: _state_used=800  budget=1200  demotions=4  over=False
lookup(A)        : _state_used=1600 budget=1200  OVER BUDGET=True  promotions=1
one more insert  : _state_used=800  budget=1200  over=False  demotions=6
```

`lookup` gets away with it — the entry is MRU, about to serve, and an insert follows in the
normal course. A DONE-path call has no such guarantee, and a demoted entry is demoted
*because* of byte pressure. Promoting it there adds bytes back under pressure and walks away.

## Rule

**Before building the fix a number asks for, check that the number's operand still exists at
the point the fix runs.** The 2448 came from a chunk-arithmetic replay with no store, no tier
and no pressure, so it counts positions that are *schedulable*, not spills that would *happen*.
Multiplying it by a survival rate nobody had measured was the whole task, and it was cheaper
than the fix.

**Report `spill_yield` and `match_yield` separately, always.** One tier moves one of them.
Reporting a single "yield" made a 0%-versus-100% difference invisible, and it reached ckl.

**A sweep sampled only at its endpoints cannot distinguish a ramp from a step.** Four slot
counts read as a threshold; single-slot steps are a straight line with a closed form. And the
cost of taking the measurement first is exact here: **two sessions each measured it once, and
the derivation needed no measurement at all.** LRU ordering gives the shape for free.

**Two sessions agreeing is not two measurements when the second one's input is the first one's
sentence.** "The tier only renames the failure" was mine, adopted on my word, and relayed
upward before anyone read `_entries_capacity`. It happened twice in one evening in both
directions — a step function derived from four points was adopted the same way. The agreement
felt like corroboration and carried no independent information.

**A verdict cited without its own qualifiers is a stronger verdict than the one that was
recorded.** The SSD reject was quoted here as "REJECT on the serve path" when its own text
limits it twice: to *this* tier, whose load path is unreachable, and explicitly not to the
block-granular store that replaces it. Quoted flat, it reads as a permanent closure of the
whole idea and would have buried this fix past the point where its reason expires. Read the
scope paragraph of any verdict before leaning on it, and carry the expiry with the citation —
a reject whose expiry conditions go unwritten becomes permanent by default, which the cited
entry's own body says it must not.

This is the fourth instance tonight of one shape: a description standing in for the authority
it describes — prose read as runtime fact, a relayed sentence read as repo state, two sessions
agreeing read as two measurements, and this. What makes this variant hard to catch is that
**the describer and the authority were the same person**: the shortened quotation was mine, so
no reader could see the gap from my text alone. The check has to be re-reading the source, not
re-reading the citation.
