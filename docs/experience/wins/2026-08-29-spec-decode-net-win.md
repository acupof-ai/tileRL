# Speculative decoding: 6.5x net LOSS -> 1.14-1.19x net win — H20, 2026-08-29

> Status: Shipped

## Context

The bundled NextN/MTP draft head (one full-attention layer, 425M params, 1% of
the trunk) drafts into the engine's existing forward: a decode row becomes a
`seq_q = 1+depth` row, so verification needed no second code path. Goodput was
right immediately — 1.87 committed tokens per trunk forward — and throughput
was still 6.5x WORSE than not speculating, because a single draft step cost
more than the entire 64-layer trunk tick.

## What Worked

**Serving the draft head the way the trunk is served.** `Backend.linear`'s
generic dense path runs at ~30 GB/s; the trunk never touches it because its
weights are fp4/fp8 and dispatch to kernels. Measured, same shape:

| path | ms/call |
|---|---:|
| `backend.linear`, bf16 dense | 9.7 |
| `linear[fp8-gemv]` (trunk) | 0.13 |

`reference.quant_fp8` inverts the existing `dequant_fp8`, `Model._linear` picks
`.w8`/`.wscale` up on its own, and `fc` moved to the same seam instead of
calling `backend.linear` directly.

**Deleting a duplicated `materialize`.** `build_engine` re-bound
`draft.params` to the dict `materialize` returns, which orphaned
`DraftHead.layers` — the `Model` that actually runs the projections — on the
original bf16 dict. The head was quantized correctly and the quantized tensors
were never read. This one line made three consecutive real fixes look like
no-ops.

Measured (H20, GPU 7 idle, 64 layers, 15 timed ticks, `scripts/bench_batch_decode.py`):

| arm | B=1 ms/tick | B=1 tok/s | B=8 ms/tick | B=8 aggregate tok/s |
|---|---:|---:|---:|---:|
| baseline | 64.4 | 15.5 | 96.9 | 82.6 |
| depth 1 | 105.9 | 15.1 | 319.2 | 38.4 |
| **depth 2** | 106.1 | **17.6 (1.14x)** | 151.8 | **91.8 (1.11x)** |
| **depth 4** | 119.9 | 15.6 | 162.7 | **97.9 (1.19x)** |

Acceptance is unchanged by the fp8 quantization (depth 2: 43.3% at B=1, 45.4%
at B=8), so the head's quality survives it. tok/tick saturates at 1.87 (B=1)
and 1.99 (B=8); `verify_lens` picks the depth per row from the draft's own
softmax probability, since this checkpoint ships no confidence head.

The B=8 depth-1 row (319 ms) is out of family with its neighbours and is not
explained; treat it as unmeasured rather than as a result.

## Rule

A perf fix that measures as a no-op is a fix that did not run. Three in a row
reading the same means something upstream is discarding them — check that the
object you mutated is the object the hot path reads, before writing a fourth.
