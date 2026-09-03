---
question: Why does an engine with `decode_graph=True` refuse a request while a slot looks free?
status: found while building the toggle matrix; not fixed here
source: H20 sm90 GPU 7, tilelang 0.1.13, 27B NVFP4; read on origin/main
---

# `decode_graph=True` permanently consumes one state slot and one KV block

The first decode tick that does not fill a graph bucket allocates a padding row
— `_pad_slot` and `_pad_block` at `engine.py:678` — and never frees it. The
allocation is per-engine and permanent, not per-tick.

So an engine built with `num_slots=N` serves `N-1` concurrent requests. The
`N`-th `submit` raises `LinearStatePool exhausted` rather than queueing, which
is the opposite of what the seam promises: `submit`/`poll` is supposed to admit
work and let `StepLimits` pace it.

## Where it does and does not bite

- **`cli.py serve`** has slack — 16 slots against `max_batch` 8 — so it has not
  been observed.
- **The training paths do not hit it** only because `grpo_loop` and self-OPD
  refuse an engine with the decode graph on, for an unrelated reason
  (on-policy sampling).
- **A measurement harness at `num_slots == max_batch` hits it immediately.**
  That is how it was found: the second arm of a two-arm run died on `submit`.

## Why it is worth a fix and not a note

The failure is a raise, not a hang or a wrong number, so it is loud. But it is
loud in the wrong place: the caller sees a pool-exhaustion error while a slot
census says one is free, and the slot is held by the graph machinery rather
than by any request. Anyone sizing a pool from `max_batch` will size it one too
small and find out in production.

## Rule

A permanent allocation made inside a per-tick path needs to be made where its
lifetime is visible — at capture, next to the pool sizing — or the pool's own
accounting has to know about it. A reserved row that looks like a request's row
is a slot census that lies.
