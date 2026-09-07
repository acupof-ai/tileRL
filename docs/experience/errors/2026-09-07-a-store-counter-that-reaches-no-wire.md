# A store counter that reaches no wire: `blocks_freed` was published into a dict `_build_stats` drops — 2026-09-07

**Date:** 2026-09-07
**Arch:** sm70 (Tesla V100-SXM2-32GB), verified against the live serving child on `3401476`
**Task:** row 47, following #221

## Context

#221 added `blocks_freed` to `PrefixStore` to make one failure visible: eviction that drops
entries and frees nothing, because `free_block` is a refcount decrement and a live request's
retain pins the blocks. The counter is measured in `_drop` from the free list before and
after, it has a test, and it is correct.

Its PR body said "`blocks_freed` joins the stats, measured in `_drop` from the free list, so
this cannot hide again", and I told a peer the field was readable. Then I read it:

```
curl /health | jq '.stats | keys'   ->  'blocks_freed' in stats == False
```

Not on the wire at all. The sentence was true of `kv_cache.py` and false of the endpoint,
and I wrote it after reading the module that publishes the field rather than the outermost
consumer. One `curl` was the whole check.

## Root Cause

`Engine._build_stats` does not forward the store's dict. It names the keys it wants and adds
one prefix splat:

```python
"prefix_evictions": store["evictions"],
"prefix_state_bytes": store["state_bytes"],
"prefix_state_bytes_budget": store.get("state_bytes_budget", 0),
**{k: v for k, v in store.items() if k.startswith(("dram_", "ssd_"))},
"prefix_demoted": store.get("demoted", 0),
```

A key matching neither the named set nor the prefix rule is dropped silently. So a counter
named `dram_anything` would have arrived automatically, and `blocks_freed` did not. Nothing
failed: the store's test asserts the store's dict, and the server tests assert the keys they
already knew about.

**Three keys were dropped, not one.** Enumerating what the store publishes against the live
endpoint:

| store key | `/health` | |
|---|---|---|
| `blocks_freed` | — | **dropped** |
| `entries` | — | **dropped** |
| `capacity` | — | **dropped** |
| `evictions` | `prefix_evictions` | forwarded |
| `state_bytes` | `prefix_state_bytes` | forwarded |
| `state_bytes_budget` | `prefix_state_bytes_budget` | forwarded |
| `demoted` | `prefix_demoted` | forwarded |
| `hits` / `misses` | see below | **not forwarded, and a name check says otherwise** |

`entries` and `capacity` are the store's fill and its ceiling — the two numbers that answer
"is the prefix cache full", which nothing else answers.

## The naming coincidence that defeats the obvious test

`/health` carries `prefix_hits`. It does **not** come from the store: `engine.py` publishes
`self._prefix_hits`, the engine's own counter, incremented inside `_admit`. The store keeps
its own `hits`/`misses` and those are dropped too.

So `assert f"prefix_{k}" in health` is **true for `hits` for the wrong reason** — two
counters for one concept, one observable, and the shared name hides it. A name-based seam
test passes while the store's counter goes nowhere.

The two counters also mean different things, and since #221 they diverge: the store counts
**lookups**, and `_match_prefix` now runs once per admission attempt, so a request that
waits inflates them. The engine's counts **admissions**. `prefix_hits` is the one a reader
wants.

## Fix

Forward the three, and make the drop deliberate rather than accidental:

- `prefix_blocks_freed`, `prefix_entries`, `prefix_capacity` added to `_build_stats`.
- `_STORE_STATS_INTERNAL = ("hits", "misses")` with the reason in a comment: the store's pair
  stays internal because it counts lookups, and because it has four in-tree readers —
  `tests/test_kv.py`, `tests/test_e2e.py`, `scripts/probe_mmlu_concurrency.py`,
  `scripts/bench_chat_reuse.py` — one of which feeds `bench_harness.py`'s `kv-reuse` gate
  whose recorded values ARE that definition. Redefining that cell inside an observability fix
  is the bucketing-key trap, so unification is its own task.

Additive only: no key renamed, no key removed.

## Gate

`test_every_key_the_store_publishes_reaches_health_or_is_named_as_dropped` asserts the
**route**, not the name: for each key the store publishes, either `/health` carries the same
**value** under `prefix_<k>`, or it matches the `dram_`/`ssd_` splat, or it is named in
`_STORE_STATS_INTERNAL`. Plus `test_blocks_freed_moves_on_the_wire_when_the_store_frees_a_block`,
which drives a real eviction through `_drop` and asserts the endpoint's number moves with the
store's.

Three controls, each run separately:

| mutation | result |
|---|---|
| the `prefix_blocks_freed` line removed | `the store publishes ['blocks_freed'] and nothing forwards them` |
| `"prefix_blocks_freed": 0` hardcoded instead of the store's value | `['blocks_freed=2 vs prefix_blocks_freed=0']` |
| a key both forwarded and named in the drop list | see below |

**Two of the three controls found holes in the test rather than confirming it**, which is the
part worth keeping:

**The first control reded for the wrong reason.** Both tests failed with
`KeyError: 'prefix_blocks_freed'` — a crash, not a finding. A reviewer reads that as a broken
test, and it would have hidden a genuine second defect behind the first. Changed to an
explicit membership assert that reports "not on the wire".

**The third control passed.** Adding `entries` to the drop list while it was also forwarded
raised nothing, because the test asked "routed somehow" and both branches were routes. Fixing
that surfaced the `hits` problem: the both-routes check flagged `hits`/`misses` as forwarded
AND internal, which is exactly the naming coincidence above — and that is what forced the test
onto values instead of names. **A name check cannot express this seam at all.**

And the value check needed a non-trivial fixture: every counter is 0 on a fresh engine, so
`health[wire] == store_vals[k]` passes against a hardcoded `0`. The test now inserts, looks up
and clears first, and **asserts the counters are non-zero before comparing them** — without
that line the hardcoded-zero control passes.

`450 passed, 14 skipped`. `ruff check` clean.

## Rule

**Read a new counter from the outermost consumer before claiming it is observable**, never
from the module that publishes it. One `curl`.

**And gate the seam by value, not by name.** A name test cannot tell a forwarded key from a
hardcoded zero, nor a store key from an unrelated engine key that happens to share the name.
Where two layers keep counters for one concept, the shared name is the thing that hides the
break.

Related: the inbound twin is a flag that parses and never reaches the engine — same seam,
opposite direction, and neither test catches the other.
