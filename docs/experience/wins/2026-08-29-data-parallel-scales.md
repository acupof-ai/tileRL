# Data parallelism scales 7.54x on 8 H20s — measured before writing any of it

> Status: Measured. The wrapper is the next unit of work; the ceiling is known.

## Context

The 27B in NVFP4 is ~23 GB and one H20 has 96 GB, so the model fits on a single
card. That single fact sets the multi-GPU priority: **tensor and pipeline
parallelism are not needed for capacity here** — they buy latency, and pay
communication for it. Data parallelism buys aggregate throughput and pays
nothing but memory the cards already have.

Before writing a line of it, the ceiling is measurable: N independent processes
are the upper bound on any in-process data-parallel wrapper, since the wrapper
can only add contention (one GIL, one CUDA context switch per tick).

## The Measurement

`scripts/pod_fan.sh`, the same B=8 graph-captured decode bench on all 8 cards
simultaneously:

| | aggregate tok/s |
|---|---:|
| one card, alone | 135.4 |
| each of 8 cards, concurrent | 124.3 - 130.1 |
| **8-card total** | **1021** |

Per-card cost of full occupancy: ~5%. **7.54x on 8 cards.**

Nothing shared saturates — not host dispatch, not PCIe, not memory bandwidth.
That was worth confirming rather than assuming: the 971 synchronous pageable
H2D copies fixed earlier today were exactly the kind of per-tick host work that
would have serialised eight processes against each other.

## What This Decides

- **DP first**, and it is an engine-level wrapper: N engines, one per device,
  requests routed to the shortest queue. No kernel changes, and the
  submit/poll/step seam does not move.
- **TP second**, for single-request latency (B=1 decode is 92.3 tok/s; TP is the
  only lever that attacks it). GDN shards cleanly by value head — 48 divides by
  2, 3, 4, 6, 8.
- **CP third**, for the 128K/256K rows. The gated-delta recurrence is sequential
  in T, so context parallelism needs the per-chunk state hand-off between ranks.
- **PP not at all** while the model fits on one card: it would add pipeline
  bubbles to buy capacity nobody needs.

## Rule

The upper bound on a parallel implementation is usually measurable without
writing it. N processes bound any in-process scheme; a single kernel's grid
bounds any schedule change. Measure the bound first — it costs one run and it
tells you whether the implementation is worth its risk.
