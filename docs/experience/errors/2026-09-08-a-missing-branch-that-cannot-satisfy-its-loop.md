# A missing branch is only a defect if the branch could satisfy its loop — 2026-09-08

> Status: **closed 2026-09-08.** Nothing to fix in `src/`. Two claims withdrawn, one of them
> mine, made while reviewing the other. The correct reading is in
> [OPEN.md](../OPEN.md) under Struck and in PR #279's comments.

## Context

`PrefixStore` has a host snapshot tier. Three loops in it shed load:

```
kv_cache.py:1225  while len(self._by_id) > self.capacity or self._state_used > self.state_bytes:
                      if self._dram is not None and len(self._by_id) <= self.capacity \
                              and self._demote_one():
                          continue
                      self._evict_one()

kv_cache.py:1330  def evict_until_free(self, blocks: int) -> None:
                      while self._pool.free_blocks < blocks and self._by_id:
                          self._evict_one()
```

Two claims were filed against them, six days apart, both saying a demote branch was missing
or unreachable:

1. **The count guard is conflated** — `len(self._by_id) <= self.capacity` sits inside a loop
   entered when `len > capacity`, so it "can never be true" and the store evicts where it
   could demote. On [OPEN.md](../OPEN.md) as a latent defect.
2. **`evict_until_free` has no demote branch** — it calls `_evict_one` directly, so block
   pressure drives into the one path that cannot use the tier.

I wrote the second one into a peer's PR review as a correction to my own record, while
reviewing and confirming the refutation of the first.

## Root Cause

Neither is a defect, and both fail for the *same* reason. A loop sheds load until its own
condition is false. A demote branch helps only if a demote can make that condition false.

| loop | condition | does a demote shrink the term? |
|---|---|---|
| `insert`'s shed loop, byte term | `_state_used > state_bytes` | **yes** — the snapshot's bytes leave |
| `insert`'s shed loop, count term | `len(_by_id) > capacity` | no — `_demote_one` sets `entry.state, entry.demoted = None, True` and **leaves the entry in `_by_id`** |
| `evict_until_free`, block term | `_pool.free_blocks < blocks` | no — a demote moves snapshot bytes and touches no block |

`_demote_one` is a byte-axis tool. The other two loops have no byte term in them, so in both
a demote branch would spin: write to the tier, re-test an unchanged term, write again, and
end up evicting anyway — one wasted tier write plus a `forget` per entry. The existing code
is not conservative, it is the only correct shape.

**That is structural, and the measurements below only confirm it.** `_demote_one` subtracts
from `_state_used`, sets `entry.state = None` and `entry.demoted = True`, and touches neither
the block pool nor `_by_id`. So it *cannot* free a block or shrink the count, whatever any run
happens to show. A measurement says a thing did not happen once; the code says it cannot. The
numbers are here because a claim of this shape should not rest on a reading alone — the first
claim was a reading, self-consistent, and wrong.

Byte pressure with the entry count exactly at capacity, which is the boundary the guard sits
on:

| pressure | entries | demoted | evictions | `dram.demotions` |
|---|---:|---:|---:|---:|
| byte only — `capacity=1000`, `state_bytes=2500`, 9 published | 9 | 6 | **0** | 6 |
| count only — `capacity=5`, `state_bytes=1 TiB`, 9 published | 5 | 0 | 4 | 0 |
| **boundary** — `capacity=9`, `state_bytes=2500`, 9 published | 9 | 6 | **0** | 6 |
| both — `capacity=6`, `state_bytes=2500`, 12 published | 6 | 3 | 6 | 9 |

The boundary row is the negative control the first claim needed: `len == capacity` satisfies
`<=`, so the guard is true at the exact count where "can never be true" is most load-bearing,
and 6 demotions with 0 evictions come out.

And for the block axis, a 64-block pool with the byte term driving demotes:

```
reclaimable_blocks()=6   free_blocks=58
one forced _demote_one(): returned True, demotions 3->4, free_blocks 58->58
delta free_blocks from a demote = 0
```

## Why reviewing the first claim did not stop the second

The refutation I reproduced was *about the count term*. I confirmed it, then reached for a
second claim on a third term and applied the opposite reasoning to it without noticing they
were the same question. The specific slip: I quoted a paragraph from
[the four-mechanisms entry](2026-09-08-four-mechanisms-one-regression-and-a-copied-flag.md)
and stopped reading before the sentence that answers it —

> **And that asymmetry is not an oversight.** At `:684` the caller needs `needed` blocks
> *now* to admit a row, and demotion frees **zero** blocks.

— which is under the very heading I was citing, **The locality: `evict_until_free` is the one
eviction path with no demote branch**.

## What survives

The run that produced both claims does hold a real defect, on neither axis: the tier converts
byte pressure into block pressure. Enabling it took peak occupancy **45.6% → 99.1%** on the
same pool and workload, because a demote keeps the entry and therefore keeps its blocks. The
block path then evicts demoted entries, `_drop` calls `_dram.forget`, and the host
copies are orphaned — **103 demotions, 1864 ms, 0 promotions**, 5.9 GiB of tier budget unused.
That is a sizing defect: retain fewer blocks, or size the pool against the tier's retention.
It is the [OPEN.md](../OPEN.md) row that stays.

## Rule

Before filing a missing branch as a defect, name the loop condition the branch would have to
falsify, and check that the branch's effect appears in that condition. A branch that cannot
satisfy its own loop is not a fix that was forgotten; it is a fix that does not exist.

Corollary, the part that cost the second instance: confirming a refutation on one term is not
license to reason about another. Both claims here were one question — *which axis does this
tool act on* — asked about three terms, and the answer differs per term. Reviewing one term
teaches nothing about the next unless the term is named.

Second corollary, on how the wrong claim got out: I read the paragraph that supported it and
stopped at the paragraph that refuted it. **Evidence that matches the expectation ends the
search**, which is the same failure whether the evidence is prose, a number, or a grep hit.
The stopping rule cannot be "I found something consistent". For a claim about code it is: read
the whole section, and name the mechanism by which the claim could be false.
