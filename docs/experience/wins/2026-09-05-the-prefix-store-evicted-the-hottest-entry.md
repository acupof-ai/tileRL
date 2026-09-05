# The prefix store evicted the entry every workload re-reads — V100 sm70, 2026-09-05

> Status: Shipped (CPU-gated). Hit-rate on the two real workload shapes is
> **pending-remote** — the shapes are named below, not measured.

## Context

Designing KV tiering, which needs a demotion order. Read `PrefixStore` and found
the shipped policy is **FIFO**, not LRU: `_fifo: deque` with `popleft`, and
`lookup` did not touch recency at all.

FIFO is the worst possible order for both workloads this server actually has:

- **A chat client** resends the whole conversation each turn, so the prefix grows
  strictly and the entry wanted on turn N is the one inserted on turn 1.
- **An RL rollout group** shares one prompt across 8 samples, so the prompt's
  entry is read 8 times and inserted once.

In both, **the hottest entry is the oldest**, which is exactly what FIFO evicts.

## What Worked

LRU by making recency *be* the iteration order, so the change deletes a field
rather than adding one:

```python
self._by_id: OrderedDict[int, _Entry] = OrderedDict()   # _fifo removed entirely
lookup hit:  self._by_id.move_to_end(e.eid)
evict:       _, entry = self._by_id.popitem(last=False)
```

`_fifo` is gone; `evict_until_free` and `clear` loop on `self._by_id`. Net one
fewer structure to keep in step.

Measured on the re-read shape — 4-entry capacity, one conversation re-read for 6
turns while unrelated traffic arrives each turn:

| policy | hits | misses |
|---|---:|---:|
| FIFO (before) | **0** | 6 |
| LRU (after) | **6** | 0 |

Negative control: deleting the `move_to_end` line (which is exactly FIFO
behaviour) makes the new test **FAIL**, with `__pycache__` cleared before the run.
The test asserts the **hit count**, not the policy's implementation, so it also
catches `_evict_one` drifting off the LRU end.

## The pre-existing inert test, which is the more useful finding

`test_prefix_hit_survives_evicting_its_own_entry` guards a real hazard: `submit`'s
own `evict_until_free` can evict the entry it just matched, so the snapshot must
be read before that call, not after. **It does not catch that bug, and it did not
before this change either.**

Probed directly rather than inferred:

```
free before submit: 1   entries: 2
matched: 32  hit_blocks: [7, 6]   needed = 2
after evict_until_free:  entries 2 -> 1,  evictions 1,  free 1 -> 3
re-match AFTER eviction -> matched: 32,  snap is None: False    <- still resident
```

The victim is the **decoy**, not the matched entry. The matched entry's 2 blocks
are already retained by this request, so `free_block` drops refcount 2→1 and
frees **nothing**; the decoy's 2 blocks satisfy `needed=2` and the loop stops.
Two separate mutations moving the snapshot read after eviction both **survived**.

The assertion is `evictions >= 1` — that eviction *happened*, not that it reached
the matched entry. Those are different claims and only the second one is the
hazard.

**Left exactly as it was**, with a `# ponytail:` comment recording what it cannot
catch. Not repaired here because making it bite requires the matched entry to be
the only source of free blocks, and that configuration raises "insufficient KV
blocks" before reaching the snapshot — a fixture problem, not a one-line fix.

**And it is untouched by FIFO→LRU**: nothing in it calls `lookup`, and with no
hits recorded **LRU order is insertion order**, so the first-published entry is
still the first victim.

## Rule

A test named for a hazard can assert a weaker claim than the hazard — "eviction
ran" is not "eviction reached this entry." Mutate the exact bug before trusting
the name.

And when a policy change appears to break a neighbouring test, check whether that
test ever worked: I rewrote this one twice on the assumption LRU had broken it,
and the second rewrite failed for an unrelated reason, before probing the actual
eviction and finding it had never exercised its own hazard.

## Results

| date | commit | machine | target | workload shape | FIFO hits | LRU hits |
|---|---|---|---|---|---:|---:|
| 2026-09-05 | (this) | Mac (cpu) | cpu | re-read prefix, 6 turns, capacity 4 | 0/6 | **6/6** |

Suite: 293 passed, 14 skipped; `ruff check` clean.

**Not claimed:** any figure on the 27B, or a hit-rate change under real traffic.
The two shapes above are what to measure, at `--max-ctx 32768` — the 8x context
that landed today is a flag, and folding it into a cache-policy result would
credit LRU with it.

`state_bytes` accounting is unchanged by design: LRU reorders which entry leaves,
not what an entry costs. Both eviction triggers (`len(self._by_id) > capacity`,
`_state_used > state_bytes`) and the `+=`/`-=` of `entry.nbytes` are byte-identical.
