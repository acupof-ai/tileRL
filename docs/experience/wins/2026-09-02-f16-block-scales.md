# f16 block scales — V100 (sm70), 2026-09-02

> Status: Shipped

## Context

The NVFP4 block-scale plane is one f32 per 32 weights: **3.20 GB of the 16.04 GB
a dense decode token streams**. Nothing else structural was left to remove — the
weights are already 4-bit — so this was the last available cut to the
denominator of the byte roofline.

The line had been rejected once, with "relerr 2.0-21.3, the implementation is
wrong". That verdict was the metric, not the kernel (see below).

Workload: `bench_ctx_decode.py`, dense B=1 greedy decode, decode ticks only, one
job on the host.

## What Worked

Three changes, one mechanism:

1. **`sh` factory arg on `make_linear_fp4_gemv_sm70_m`** — annotates `Scale` as
   f16. The scale still reaches the tile extern as f32, so the dequant math is
   byte-identical; only the global-memory plane narrows.
2. **`Backend.materialize` narrows the plane once** on sm70, beside the existing
   twiddle pass. Not per call: a cast inside `linear_fp4` would reallocate 1.6 GB
   every token and stream both copies.
3. **`save_hf` widens it back**, because the format's scale is f32 and a
   checkpoint written from a V100 must load anywhere.

Representability was measured on the real checkpoint before any kernel work
(`scripts/check_scale_f16.py`, 252M scale values): magnitudes span
2.5e-03..1.99, nowhere near f16's limits, and the worst round-trip is
**3.24e-04** relative — 31× inside the 1e-2 parity gate. bf16 also fits
(2.59e-03) but buys nothing extra; **e4m3 is dead at 3.06e-01** despite halving
the plane again.

| ctx | f32 scales | f16 scales | gain |
|---:|---:|---:|---:|
| 32 | 38.7 | **41.6** | +7.5% |
| 512 | 38.2 | 41.0 | +7.3% |
| 1024 | 37.7 | 40.5 | +7.4% |
| 2048 | 36.9 | 39.5 | +7.0% |
| 4096 | 35.3 | **37.6** | +6.5% |

Bytes predicted 1.111×; measured 1.065× at 4096 — **60% of the traffic saving
converted to time**, consistent with the GEMV running at 84% of its own byte
roofline. Device memory fell 31.6 → 29.0 GB, the 1.6 GB the plane saves, which is
independent confirmation the narrowing actually happened.

Roofline moves 56.1 → 62.3 tok/s, so dense at 4096 is now 60% of it.

## The rejection was the yardstick, twice

The first attempt died on `relerr` 2.0-21.3. Re-measured, the error is
**1.88e-04** and flat in N; the reported number came from
`clamp(min=1e-3)` in the denominator, which floors it and lets any near-zero
output row manufacture a huge ratio — and being a `max()` over N rows, it grows
with N by construction (0.21 at N=1024, 21.8 at N=248320). `benchkit.relerr` had
the correct max-abs-normalized form the whole time.
`errors/2026-09-02-clamped-relerr-scales-with-n.md`.

## Rule

The scale plane is a fifth of an NVFP4 weight stream, not a rounding error —
check its width before optimizing the nibbles. And a parity number that varies
with N is measuring the yardstick: arithmetic error from a wrong dtype or layout
cannot depend on how many rows there are.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-02 | (this) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | — | 26.6 | 37.6 @ 4096 ctx |
| 2026-09-02 | 6332d44 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | — | 28.3 | 35.3 @ 4096 ctx |

Raw artifacts: `scripts/ab_scale_f16.py` (per-shape A/B + achieved GB/s),
`scripts/check_scale_f16.py` (storage-width survey on the checkpoint),
`scripts/bench_ctx_decode.py` (the tok/s table above).
