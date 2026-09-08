# `spill_held`: the store side of the boundary fix, with no caller yet — 2026-09-08

> Bench entry for a runtime change whose measured perf delta is **zero by construction**:
> `PrefixStore.spill_held` is new public surface with **no caller under `src/`** (grep:
> `spill_held` appears in `kv_cache.py`'s definition and in `tests/test_kv.py`, nowhere else).
> The engine side is [`tilerl-48`'s half](../errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md);
> until it lands, no served tick reaches this method. Not `pending-remote` — there is nothing
> a card would measure.

## Context

[The boundary entry](../errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md) closes
`budget == 512` and stays `Status: open` for the rest: 2448 of 35756 prompts in the default
504–512 window still spill nothing, because `_last_prefill_boundary` takes `n` where the
boundary is `f(n, budget)`. Four fixes were measured and rejected. The one that survived —
capture the snapshot at the boundary and spill it later — needs the store to accept a spill
for an entry it already holds, which it did not.

## What Worked

`insert` refuses a duplicate before its write-through (`kv_cache.py`, the `for e in
self._entries.get(h, ())` loop), so re-offering held tokens with `spill=True` returns False
having touched nothing. Measured on the real store with a real `KvTier`: the tier's `offered`
stays **0** — not a refusal, no counter moves at all, indistinguishable from a spill that
happened. A fresh entry with `spill=True` gives `resident=True, offered=1`, so the harness
does spill when the path is reachable.

`spill_held(tokens) -> str` is that path. It returns `spilled` or the reason it did not:
`no-tier`, `no-entry`, `demoted`, `no-state`, `resident`, `refused`.

**A string rather than a bool, because two failure modes are silent.** Between the boundary
publish and DONE the entry can lose its snapshot without losing anything a caller would check:

| pressure | entry | `state` | blocks | how the caller sees it |
|---|---|---|---|---|
| DRAM tier demote (`state_bytes=1200`, 5 publishes) | present | **None**, `demoted=True` | refcount 1 | **silent** |
| count pressure (`capacity=3`, 6 publishes) | **gone** | — | — | loud |
| no pressure (control) | present | held | refcount 1 | — |

A caller written as "the entry is still there, so spill it" would do nothing on any config
with a DRAM tier — the same silent no-op this whole path exists to remove. The blocks are the
one thing that cannot vanish (`insert` retains them; refcount 1 in both pressure cases,
measured, not reasoned).

## Why `demoted` does not promote-then-spill

It could: `DramSnapshots.promote` pops the host copy back and the store owns the snapshot
again. It must not, and the reason is mechanical rather than a cost judgement.

`lookup`'s promote does `self._state_used += e.nbytes` and returns **without re-entering** the
`while len > capacity or _state_used > state_bytes` loop at the end of `insert`. Measured:

```
after 5 publishes: _state_used=800  budget=1200  demotions=4  over=False
lookup(A)        : _state_used=1600 budget=1200  OVER BUDGET=True  promotions=1
one more insert  : _state_used=800  budget=1200  over=False  demotions=6
```

So a promote leaves the store over budget and the *next insert* rebalances it. `lookup` gets
away with that — the entry is MRU, it is about to serve a hit, and an insert follows in the
normal course. A DONE-path call has no such guarantee, and a demoted entry is demoted
*because* of byte pressure: promoting it there adds the bytes back under pressure and walks
away. The cost argument agrees but is secondary — 144 MiB at 11.52 GiB/s pinned is 12.7 ms,
and the bytes written would be a third copy of a snapshot the second turn must promote anyway.

## The gate

Two tests, both with the control inside them:

- `test_a_held_entry_reaches_the_tier_through_spill_held_not_insert` — fresh `spill=True`
  lands (`offered == 1`); a duplicate `insert` still returns False with `offered == 0`;
  `spill_held` then returns `spilled` and the tier has the bytes; a second call returns
  `resident` rather than re-sending them.
- `test_spill_held_names_every_reason_it_did_not_spill` — `no-tier`, `no-entry` (never
  published), `demoted` (asserting the fixture actually demoted and that blocks survived), and
  `no-entry` again from count eviction.

Mutation-checked: replacing `return "demoted"` with a promote-then-fall-through turns the
second test red at `demoted != spilled`, and reverting turns it green. The earlier version of
the first test asserted today's refusal and was verified red against the candidate fix that
spills inside `insert`'s duplicate branch, so both directions of that boundary are pinned.

30 passed, 6 xfailed in `tests/test_kv.py`; `ruff check` clean.

## Rule

New surface with no caller is still a runtime change and still gets an entry — the entry's job
here is to record that the perf delta is zero *because* nothing calls it, which is a fact that
expires the moment the engine half lands. An entry saying "measured zero" without saying why
would read as a measurement rather than a construction.

Second: when a method's failure modes differ in whether the caller can see them, the return
type has to carry the difference. A bool would have been shorter and would have handed the
engine half a silent no-op on exactly the configuration the tier exists for.
