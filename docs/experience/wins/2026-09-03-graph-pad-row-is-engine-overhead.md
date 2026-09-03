# The decode graph's padding row is engine overhead, not the caller's capacity — 2026-09-03

> Status: fix + CPU gate with a negative control. No hot-path arithmetic
> changed; the cost is 156.2 MiB on the 27B, computed from the config.

## Context

A captured decode tick replays at a fixed batch size, so a tick with fewer rows
than its bucket pads. The padding rows still write to both pools, so they need
a state slot and a KV block of their own. The engine took them by calling
`alloc_slot()` / `alloc_block()` on the pools the caller had sized, on the
first tick that padded, and never returned them.

So `num_slots=N` served N-1 requests. The N-th `submit` reached
`self._states.alloc_slot()` at `engine.py:366`, which has no fallback, and the
`RuntimeError` came back to the caller
([errors entry](../errors/2026-09-03-decode-graph-leaks-a-slot-and-a-block.md)).

Two things made it hard to see. The reservation is lazy, so a run served N
requests until the first tick that happened to pad, and lost one after. And the
allocation inside `_run_decode_graph` is wrapped in `try/except RuntimeError`
with a fallback to an exact-size graph, which reads like the exhaustion case is
handled — it is, but for the graph, not for `submit`.

## What Worked

The padding row is the engine's, not the caller's, so the pools are sized for
it and the reported capacity is not.

- `build_engine` allocates `num_slots + 1` and `num_blocks + 1` when the
  captured tick is on.
- `Engine.__init__` reserves the pad row up front instead of on the first tick
  that pads. A capacity that drops one request partway through a run is worse
  than one that is whole and one smaller from the start, and the up-front
  reservation is also what makes the pool sizing and the reservation agree by
  construction rather than by timing.
- `stats()` reports `slots_total` and `blocks_total` net of the pad row, so the
  numbers a caller reads are the ones it can use.
- `_run_decode_graph` no longer allocates. It checks whether the reservation
  exists and drops to an exact-size graph if not, which is the path a directly
  constructed `Engine` takes when its caller sized the pools without the spare.

`decode_graph=None` meant "on for CUDA" in one place and had to mean the same
thing in `build_engine` for the sizing to match the reservation. `_graph_on`
is that one definition; both call it.

## The cost

Computed from `qwen38_27b()`, not measured: 64 layers, 16 full-attention, 48
linear.

| | |
|---|---:|
| one state slot (f32) | 144.0 MiB |
| one conv window (f32) | 11.2 MiB |
| one KV block (bf16) | 1.0 MiB |
| **pad row** | **156.2 MiB** |
| share of the 27B serving peak (38.62 GiB) | 0.395% |

0.395% of peak buys back one concurrent request. The alternative — keep taking
the slot from the caller — costs the same 156.2 MiB and one request as well;
it just does not say so.

## The gate, and its negative control

`test_the_graphs_padding_row_is_not_taken_from_the_callers_capacity`, CPU cell.
It runs on any target because what is gated is the reservation and the
accounting, not the capture, which only CUDA does.

Asserted: with the graph on, the pools grow by one and the reported capacity
does not; the pad slot and block are held; and `num_slots` concurrent submits
all succeed, the N-th being the one that used to raise. With the graph off —
the negative control — nothing is reserved and nothing is added.

Mutating `num_slots + pad` back to `num_slots` turns it red
(`assert 3 == 3 + 1`); restoring it turns it green.

## The seam the first fix left open

`stats()` was made net of the pad row and `submit`'s KV guard was not:

```python
if self._kv.blocks_for_tokens(total + self._spec_depth) > self._kv.num_blocks:
```

`num_blocks` is now gross, so the one request sized to the whole pool passed
the guard and died on the allocation behind it —
`RuntimeError: insufficient KV blocks for request` instead of the `ValueError`
that names the cause. The same shape as the pad row itself, one level down: a
capacity check against gross capacity where the usable capacity is net.
Found in review, not by the gate, which is what a second reader is for.

Patching the one call site would have left the next one to drift, so usable
capacity has one definition — `usable_blocks` / `usable_slots` — and the guard
and `stats()` both read it. The block-table widths at `engine.py:174/182/554`
stay gross: they size a tensor, they do not answer whether a request may have
something. The draft KV plane spans the trunk's whole block space by design, so
it includes the pad block correctly.

`test_the_kv_guard_measures_usable_capacity_not_the_pool` submits a request
needing 17 blocks against 16 usable and 17 gross, with the graph-off engine as
the control that must reject it identically. Reverting the guard to
`self._kv.num_blocks` reproduces the original failure exactly: the request gets
past `submit` and raises `RuntimeError` from the allocator.

## Rule

A resource the engine needs to run is the engine's, and it is sized for. Taking
it out of the pool the caller sized converts an engine cost into a silent
capacity cut, and a lazy take converts it into a capacity cut that appears
partway through a run.

A `try/except` around an allocation says the allocation's own failure is
handled. It says nothing about the second caller of the same pool.
