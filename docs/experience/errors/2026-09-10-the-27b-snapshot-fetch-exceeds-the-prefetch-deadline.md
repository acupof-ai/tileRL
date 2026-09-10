# The 27B snapshot fetch exceeds the prefetch deadline: the spin hits its 50 ms bound on every tick and the fetch drops

## Context

OPEN row 17's remainder asked for the spin's cost on the full 27B with SSD on.
The slice bench ([wins/2026-09-10-spin-until-ready](../wins/2026-09-10-spin-until-ready-fixes-cpu-prefetch-flake.md))
measured −0.1%/−0.2%, but the spin never ran in that bench — no fetch was in
flight during decode, so `any_fetching()` was false at every tick. Those numbers
price the `any_fetching()` check itself, not the spin. The slice's `.st` is 9.8
MB (the 18 KB was the `.kv`); the full 27B snapshot is 155.2 MiB (states 144.0 +
conv_window 11.25), so a fetch spans many GIL windows and the spin actually
runs.

Measured 2026-09-10 on H20 card 0 (sm90), `scripts/probe_spin_cost_27b.py`:
warm arm spills a 128-token prompt to SSD, cold arm recovers and submits the
192-token extended prompt, timing each `step()` bucketed by `any_fetching()`.

## Root Cause

The fetch takes **117 ms** to load the 157.6 MiB snapshot
(`fetch_bytes=165281792`, `fetch_ms=117`). The deadline is
`len(tokens) / seed_rate = 192 / 2558.6 = 75 ms` (sm90 seed rate,
`engine.py:509`). The fetch is 1.56x the deadline, so it always drops
(`fetches_ready=0, fetch_drops=1, hits=0`).

The spin hits its 50 ms bound on **every tick while a fetch is in flight**
(3/3 spin ticks: 50.15, 50.20, 6447.86 ms — the last is the prefill tick,
where the spin is 50 ms of a 6.4 s forward). Quiet decode ticks (no fetch)
median 11.39 ms. The spin adds ~39 ms per tick while a fetch runs, and the
fetch still drops.

The deadline formula (`len(tokens) / seed_rate`) assumes fetch time scales with
token count. On 27B the fetch time is dominated by the constant snapshot size
(155.2 MiB), not the token count — a 192-token prompt and a 1024-token prompt
fetch the same 155.2 MiB. The seed rate (2558.6 tok/s) prices the prefill
forward, not the snapshot load.

`any_fetching()` is global, not per-request (`kv_cache.py:1100` returns the SSD
tier's flag, not the held row's). Under concurrency, an unrelated fetch makes
every tick spin ~50 ms. The B=8 −0.2% in the slice bench cannot answer this —
it had no fetch in flight.

## Fix

Not fixed in this PR — this is a measurement PR. Two directions, both named in
the original errors entry:

1. **Account for the snapshot size in the deadline.** The deadline should be
   `max(len(tokens) / seed_rate, snapshot_load_ms + margin)`. The snapshot
   load time is measurable (`ssd_fetch_ms` / `ssd_fetch_bytes` gives MiB/s).
2. **Do not abandon an in-flight fetch when the deadline expires** (the
   "alternative safety net" from
   [errors/2026-09-10-prefetch-deadline-gil-contention](2026-09-10-prefetch-deadline-gil-contention.md)):
   the 117 ms is already paid, and the result is useful for the next request
   with the same prefix. The deadline should govern whether to *start* a fetch,
   not whether to *discard finished work*.

Direction 2 is the cheaper fix and also closes the global-`any_fetching()`
cost: a fetch that is not abandoned completes, and the next tick stops spinning.

## Rule

A deadline computed from token count assumes the fetch time scales with tokens.
When the fetch loads a constant-size snapshot, the deadline must account for
the snapshot size — or the fetch always drops on a model where the snapshot is
large. The seed rate prices the forward, not the load.

## Results

| date | machine | target | model | snapshot | fetch_ms | deadline_ms | spin ticks | spin bound fires | quiet tick med |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-09-10 | H20 card 0 | sm90 | 27B-NVFP4 | 157.6 MiB | 117 | 75.0 | 3 | 3/3 | 11.39 ms |

Raw artifacts: `/work/spin27b.log` on the pod.
