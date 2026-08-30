# sm70 fp16-twiddle GEMV: decode 19.9 -> 26.4 tok/s — V100/sm70, 2026-08-30

> Status: Shipped

## Context

The sm70 fp4 GEMV decoded natural nibbles with a branch-free bit-synthesis
(~10 ops/elem, issue-bound at 28-38% roofline). The fp16-twiddle permutes the
packed bytes offline so each nibble lands in an fp16 exponent/mantissa field;
one `mul.f16x2` by 2^14 rebiases the e2m1 exponent, cutting decode to ~1.9
ops/elem. sm70 has no `cvt.rn.f16x2.f32` (sm80+) and no `mul.bf16x2`, so the
sm90 bf16-twiddle twin is dead here — this is the fp16 twin, its C extern
(`T.call_extern` + `T.import_source`) doing straight-from-global loads.

## What Worked

- **fp16-twiddle decode** (`tl_fp4_decode8_f16`): prmt + shift/mask + 4x
  `mul.f16x2` by 0x74007400 (2^14 per lane). Bit-exact vs the e2m1 LUT
  (numpy sim, 50k words; `tests/test_fp4_twiddle.py`).
- **sm70 GEMV** (`make_linear_fp4_gemv_sm70`): GROUP=4 tiles of 16 elem per
  thread, fp16 accumulate inside the 16-elem scale block, one f32
  scale-accumulate per tile. Split-K (reduce_thread=32, block_K=512).
- **Eager twiddle in `materialize`** (not lazy in `_served_fp4`): the twiddle
  allocates a same-size scratch; by the first forward the KV cache +
  activations have left no room on a 32GB card. Tagged `_tl_layout` so
  `_served_fp4`, `save_hf`, and re-materialize skip it.
- **M>1 prefill fix (the regression)**: `materialize` twiddles ALL `.wq`
  weights, but sm70 M>1 has no twiddle-aware kernel — it fell to the generic
  `linear_fp4` which decodes NATURAL nibbles, silently feeding it twiddled
  bytes. The prefill corrupted the KV/hidden, so decode output gibberish.
  The M=1 GEMV was always correct (real-weight parity passed); only M>1 was
  broken. Fix: sm70 M>1 loops the twiddle-aware M=1 GEMV per row (correct,
  M launches, prefill-only). An untwiddle-copy OOMs here (the forward's GPU
  is full — the same scratch constraint that forced eager twiddle).

## Rule

A twiddle that rewrites served bytes must be matched by EVERY kernel that
reads them, or the untwiddled path silently decodes garbage. The M=1 parity
gate was too narrow — the M>1 prefill path is a separate decode contract.
Test every M-dispatch bucket on real checkpoint weights, not just pack_fp4
synthetic ones (the micro parity passed on both, but only the e2e caught the
M>1 generic-kernel mismatch).

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-29 | (prev) | V100 | cuda/sm70 | 27B NVFP4 | — | 48.0 | 19.9 |
| 2026-08-30 | (this) | V100 | cuda/sm70 | 27B NVFP4 | — | 37.9 | 26.4 |

Steady-state B=1 decode (per-step timing, first 3 ticks skipped — the first
is 51s JIT+capture, the "548s-in-the-window trap"). Graph capture SUCCEEDS
with the C-extern kernel (no fallback warning). Correctness: "The capital of
France is" -> " Paris. The capital of Germany is Berlin. The capital of
Italy is Rome...".

Micro-benchmark (large-K GEMV): 35% -> 55% MBU. Effective HBM: 475 GB/s =
53% of the V100's 900 GB/s roofline (was 42%).

**Physics ceiling**: W4A16 = 0.75 B/param (0.5 WQ + 0.25 scale) -> ~18 GB/token
-> 900 GB/s gives ~50 tok/s max. 60 tok/s is not reachable on V100 with
W4A16; 26.4 is 53% of the 50 t/s ceiling. Closing to 50 needs ~95% MBU
(occupancy: n_partition=4 -> 1 warp/row, small-K shapes latency-bound at
~8% MBU — the next lever).

Raw artifacts: `scripts/bench_decode_steady.py`, `scripts/parity_real_weights.py`,
`scripts/bench_sm70_gemv_f16.py` (V100, GPU 0, JIT-cached).
