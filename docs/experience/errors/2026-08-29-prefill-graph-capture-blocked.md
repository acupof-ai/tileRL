# Prefill graph capture: unblocked, measured, and reverted at ~1%

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

3. Found it: `Model.forward`'s own `int(seq_q_lens.min())`, added an hour
   earlier to decide `last_only`. Reading a device tensor is a host sync and is
   illegal inside a capture. The engine already holds the per-row lengths as
   Python ints, so the caller decides now — a better shape regardless, since
   that sync was also paid on every eager forward.

## Then it captured, and it was not worth keeping

With the sync gone, capture succeeds. It buys almost nothing:

| row | eager | captured | |
|---|---:|---:|---:|
| prefill/len512 | 2109.7 | 2125.8 | 1.008x |
| prefill/len2048 | 2090.5 | 2116.5 | 1.012x |
| prefill/len8192 | 2025.8 | 1994.1 | **0.984x** |

Two rows up ~1%, one down 1.6% — inside the noise, and the ~6% of host
dispatch the change was aimed at does not appear. Prefill is GPU-bound enough
that its dispatch already overlaps; decode is not, which is why the same
capture is worth 7.9x there.

**Reverted.** A code path plus four sets of static buffers, for a net zero, is
not worth carrying. The host-sync fix stays — it is correct on its own.

## Rule

A capture that fails must fall back, not fail the tick. Three failures here
were caught by one `try/except` around construction, so every broken attempt
cost zero throughput and zero correctness.

And: graph capture pays where dispatch is exposed, not where it is merely
present. Decode is 785 launches against an 11.6 ms tick and capture is worth
7.9x; prefill is ~720 launches against a 243 ms forward and capture is worth
nothing. Count the launches against the GPU time they hide behind, not on their
own.
