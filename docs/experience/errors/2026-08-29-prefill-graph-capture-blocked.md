# Prefill graph capture: implemented, blocked by a capture-hostile op in the GDN path

## Context

Prefill ends the day at 2109.7 tok/s against its own profile's 2110 tok/s
"GPU-bound" — but end to end it had been trailing that figure by ~6%, which is
host dispatch across ~720 kernel launches per forward. A prefill chunk is as
static a shape as a decode step (the engine buckets its width to 64), so the
same `_DecodeGraph` should capture it.

## What Was Done

`_DecodeGraph` takes `width`, `last_only` and `keep`; `run()` branches on the
row's phase so a PREFILL row's ids/positions/seq_len come from its chunk.
`_finish_prefill` was extracted so the eager and captured paths share the
completion bookkeeping rather than drifting apart. Capture is attempted only
for a full bucketed chunk, at most four widths.

## Why It Does Not Work Yet

Two failures, in order:

1. `'NoneType' object has no attribute 'device'` — `keep_steps` was derived
   from the chain width, and a prefill chunk is also `W > 1`, so it asked for
   per-step recurrent state buffers its pool does not have. Fixed: both call
   sites now state what they need.
2. `CUDA error: operation failed due to a previous error during capture` —
   something in the prefill forward is capture-hostile. Prime suspect: the
   gated-delta prefill path goes through `state_gather`/`state_scatter` with
   the conv-window **parity**, and `window_snapshot`'s own docstring says
   "host sync on parity". The decode path avoids it entirely by using the
   fused, in-place `gdn_decode`.

Not chased further today. The fallback is clean — a failed capture warns and
runs eager, and prefill/accuracy/kv-reuse all read 0.998-1.000x — so this costs
nothing except the win it does not yet collect.

## Rule

A capture that fails must fall back, not fail the tick. Both failures here were
caught by the same `try/except` around construction, which is why two broken
attempts cost zero throughput and zero correctness. Build the fallback before
the optimization.
