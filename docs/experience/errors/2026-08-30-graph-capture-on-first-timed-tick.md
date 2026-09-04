# CUDA graph capture lands on the first timed decode tick, not warm-up — 2026-08-30

> Context: measuring V100 sm70 steady-state decode tok/s. A first attempt read
> 128 tokens / 548 s = 0.2 tok/s — absurd for a graph-captured tick. The 548 s
> was compilation, but it landed where the timing window could not exclude it.

## What happened

The bench did: warm-up request (8–16 tokens), then a timed request (128
tokens), timing `len(out) / total_seconds`. The assumption was that warm-up
triggers all JIT + CUDA-graph capture, so the timed request replays. It did not:
**graph capture is lazy on the FIRST DECODE TICK, and warm-up's first decode
tick is where it fired — but the total-time metric on the timed request still
absorbed it** because the *timed* request's own first tick re-entered a capture
path (a fresh `_DecodeGraph` per bucket, or the timed request being the first to
reach a given batch/width bucket). Per-step timing exposed it cleanly:

```
first tick: 540561.9 ms   <- JIT + graph capture
step 2..6:  48.0 / 48.0 / 48.0 / 48.0 / 48.6 ms   <- graph replay, dead stable
steady median = 48.0 ms => 19.9 tok/s
```

The `total/count` metric folded the 540 s first tick into 128 tokens → 0.2
tok/s. The steady replay is 19.9 tok/s — a 100× reporting error from one
compilation tick in the window.

## Root cause

`total_seconds / token_count` cannot separate a one-time compile/capture cost
from steady throughput when capture is lazy and per-bucket. Warm-up only helps
if it exercises the EXACT bucket (batch size, width) the timed run will use, and
even then a fresh graph object per request re-captures.

## Fix

Time per-step, not total: `torch.cuda.synchronize()` around each `eng.step()`,
drop the first 1–2 ticks (prefill + capture), report the **median** of the
steady ticks. `scripts/_v100_steady.py` does this. Mirrors the bench harness's
`settle_decode` (peer's `scripts/benchkit.py`), which settles to pure decode
before timing for the same reason.

## Rule

Never report decode throughput as total-time / token-count when any tick can
compile or capture. Time steady ticks individually and take the median; the
first tick is a different population (prefill / JIT / graph capture) and must be
dropped, not averaged in. A 100× error hides in one un-excluded compile tick.
This is the same green-signal-too-narrow family as
[2026-08-30-sm70-220-was-truncation-not-a-bug.md]: "128/548s" reads as
throughput but measures compilation.
