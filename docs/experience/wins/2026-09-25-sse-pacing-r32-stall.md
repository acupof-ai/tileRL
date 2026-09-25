# SSE pacing smooths the R32 refresh stall — V100 27B W1024/R32, 2026-09-25

> Status: Shipped (default-on in #831; pending remote CI/merge)

## Context

With W1024/R32 + the captured decode graph, raw SSE output stuttered. Graph
ticks emit ~1.8 tokens every ~41 ms, but every 32nd tick is an eager refresh
that takes ~218–259 ms. `scripts/probe_stream_smooth.py` measured, on a
37.6k prompt, >150 ms gaps at SSE frames 31,63,…,255 — exactly every 32
forwards (`frames == decode_forwards`, so each frame is one forward). The
metric the user sees is inter-frame gap, not engine tok/s.

## What Worked

A server-side token jitter buffer (`src/tilerl/stream_pacing.py`), SSE route
only, behind `--stream-pace` (now default on; `--no-stream-pace` disables):
buffer the first 12 tokens (~288 ms, one refresh stall of headroom), then emit
deltas at an adaptive interval = `max(24 ms hot prior, cumulative mean wall
per token)`. The fixed-timetable prior keeps the fast graph-only fill from
spending the headroom before the first refresh; the cumulative mean makes the
cadence follow slow cold/hit regimes without jerking on one stall. The tail and
every non-delta frame flush immediately — only send timing changes, the SSE
envelope/fields/order/usage are byte-identical, `/ws` is untouched.

Device A/B on V100 (ThinkingCap-orig, W1024/R32/q1/cold8g; off vs on differ
only by the flag), same 37.6k prompt streamed cold / prefix-hit / hot:

| regime | pace | p50 | p90 | p99 | max gap ms |
|---|---|---|---|---|---|
| cold miss | off | 41 | 112 | 282 | 2473 |
| cold miss | on  | 41 | 42 | 243 | 289 |
| hot steady | off | 41 | 127 | 226 | 1665 |
| hot steady | on  | 41 | 42 | 211 | 242 |

Worst inter-frame gap drops from ~1.7–2.5 s to ~0.24–0.29 s; p90 collapses to
the 42 ms graph tick.

The prefix-hit regime stayed rough (max ~1.56 s): that path promotes shared
pages synchronously per page (perf1 #832) and produces non-periodic stalls
larger than the 12-token headroom. Pacing cannot smooth a producer that slow
without adding equal latency — the gap passes through. #832 is expected to
bring that regime to the hot-steady level; no pacer change needed.

A measurement trap: do NOT quote `tokens/(wall-ttft)` across arms as speed. The
paced TTFT already contains the headroom fill's worth of decode, so that
denominator is shorter on the paced arm and the rate reads falsely higher
(plus the arms ran sequentially, cold vs warmed cold tier). Pacing only
defers sends; report frame gaps, first-token time, and request-start-to-last-
token wall rate — not a decode-only tok/s that changes definition per arm.

## Rule

A periodic engine stall is smoothed at the SSE boundary with a depth-sized
jitter buffer paced at the adaptive long-run rate; it cannot (and must not try
to) hide a stall deeper than the headroom that originates in slow production.
End-to-end probes report user-visible frame timing; kernel/step speed uses
CUDA-event/step probes, never SSE-derived tok/s.

## Results

- Worst frame gap (cold/hot): 1.7–2.5 s → 0.24–0.29 s; p90 → 42 ms.
- CPU gates: synthetic periodic-stall and 47 ms/token slow regimes with a
  virtual clock (p99/max < 60 ms, order/count preserved, no trailing wait,
  pacing-off negative control stays red); SSE fidelity on/off identical,
  default-on asserted; LAYERS/frozen-surface updated.

## Open follow-up (does not block merge)

The A/B ran **off first, on second**, so the on arm had a warmer cold/SSD
tier — an order/position confound the frame-gap table does not control for.
Next card window: rerun the **hot-steady** regime with the order reversed
(on first, off second) and append the numbers here.

