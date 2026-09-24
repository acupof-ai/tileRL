# The refresh-phase reset: served rate after a fresh request — V100 sm70, 2026-09-24

> Status: `pending-remote` — the device table below is filled in by the cutover
> window; the runtime change itself is landed.

## What changed

`SparseRuntime.ticks_since_refresh` is reset at admission when no other decode row
is in flight, so a new request always starts its refresh cadence at phase 0
instead of inheriting the phase the previous requests left. See
[errors/2026-09-24-refresh-phase-inherited-across-requests.md](../errors/2026-09-24-refresh-phase-inherited-across-requests.md)
for the defect and the gate.

## Why this is a bench entry

It is a runtime change on the serving decode path, so it is priced even though the
motivation is correctness. The reset moves *which* decode ticks are eager; it does
not change how many there are. Over a long single-request run the eager share is
`1 / SPARSE_REFRESH_TICKS` either way, so the expected effect on steady-state
throughput is **zero, and that is the claim to test** — a measured difference
would mean the reset is doing something other than what it says.

Two effects are plausible and are what the device table is for:

1. **Steady-state rate** — expected unchanged. A shift here means the eager ticks
   are not uniformly distributed in cost, or that the reset retimes them onto
   cheaper/more expensive ticks.
2. **First-token / early-request rate** — the reset moves the first eager tick to a
   fixed offset from admission. On a *short* request (fewer than 8 decode ticks)
   the request may now see zero or one eager tick rather than a phase-dependent
   count, which changes total work per short request rather than per tick.

## What is measured, and how

Device side, one V100 sm70, the same window that supplies the cutover numbers:

| quantity | how | status |
|---|---|---|
| steady-state effective tok/s, reset arm | warm window on the served 32k prompts | pending-remote |
| steady-state effective tok/s, pre-reset arm | same window, reset disabled | pending-remote |
| eager tick share | `eager_tick_attribution.refresh` ÷ decode ticks | pending-remote |
| per-request first-decode phase | tracer on both paths (`build_rows` + `run_decode_graph`) | measured: 3, 6, 0, 0, 0, 0, 0, 0 before the fix |

The pre-reset arm is not a history claim: it is the same tree with the reset
removed, on the same machine, in the same window. A rate compared against a number
from an earlier window would not be evidence about this change.

## Limit carried from the errors entry

Any same-service comparison on the pre-reset tree is contaminated by the phase the
arms enter with — measured content differences up to 3x in length (879 vs 286
characters) from the phase alone. The device table must state, per arm, the entry
phase, or run each arm on a fresh service. This is the same confound the change
removes, and it is why the bench entry is pending rather than quoted now.
