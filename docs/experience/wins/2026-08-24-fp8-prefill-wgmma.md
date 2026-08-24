# fp8 prefill path (e4m3 activations + fp4->e4m3 WGMMA) — sm90, 2026-08-24

> Status: Shipped

## Context

The prefill GEMMs (`linear_fp4`, all projections re-packed to tileRL's e2m1fn
fp4) were 83% of the slice prefill tick after the GDN chunk kernel landed.
bf16 WGMMA tops out at 148 TFLOPS on H20 → 2700 tok/s theoretical for the
27B, so 3800 tok/s is unreachable in bf16; fp8 (296 TFLOPS → 5500 tok/s
theoretical) is required. The checkpoint ships `input_global_scale`
(activation quant scales) but the loader re-packs weights with its own
per-16-block scale and discards them.

## What Worked

**fp8 WGMMA with on-the-fly fp4→e4m3 dequant.** Two new sm90 kernels:

- `quant_fp8` — per-token dynamic activation quant: bf16 → e4m3, one scale per
  row (`448 / row_absmax`), shared-memory max reduction.
- `linear_fp4_fp8` — e4m3 XQ @ e4m3 W (fp8 WGMMA, f32 accumulate). The fp4
  weight dequant stays in the K-loop: e2m1fn nibble → fp32 (integer fast
  decode) → `*WScale` → cast to e4m3. Per-token activation dequant (1/scale)
  in the epilogue. Direct WGMMA accumulate into `C_local` — no temp fragment.

**pack_fp4 scale granularity per-16 → per-32.** The fp8 WGMMA K-tile is 32.
A per-16 weight scale forces 2 scales per tile, which requires either a
requant (e4m3 cast of `grid*scale`, systematic rounding bias that does not
average down) or a per-tile temp fragment (manual scale-accumulate breaks the
WGMMA pipeline, 2x slower). Per-32 gives one scale per tile, so the requant
is a single cast and the MMA accumulates directly. The bf16 MMA and GEMV
paths read the same per-32 scale.

**Tile shape 128×128×64, num_stages=3** (the fp8 example's config). At
64×128×32 the kernel was dequant-bound (1.08x over bf16); the larger tile
amortizes the e4m3 cast over 2 WGMMA steps and doubles the output tile.

Measured on the 2-GDN-layer slice (M=512 prefill, H20, contended pod):

| arm | linear_fp4 ms | GPU sum ms | slice tok/s | extrapolated tok/s |
|---|---:|---:|---:|---:|
| bf16 WGMMA | 48.73 | 58.83 | 6021 | 268 |
| fp8 WGMMA | 29.25 | 39.31 | 7839 | 399 |

Per-linear speedup 1.2–1.7x (47–64 TFLOP/s, 16–22% of fp8 peak). The
extrapolated 399 tok/s is 9.5x off the 3800 target — the gap is the e4m3
dequant cast in the K-loop (the kernel is dequant-bound, not compute-bound;
production fp8 GEMMs precompute fp8 weights and hit 60–80% of peak).

## Precision

e4m3's ~2% multiplicative quant error does **not** average down over K (it is
per-element relative error, not zero-mean additive). The fp8 parity test
gates against an identical-quant torch reference (same per-token e4m3 quant +
same requant), not the f32 reference — the kernel is correct to <1% vs the
same quant; the ~2% vs f32 is the e4m3 format floor. The e2m1fn weight grid
is an exact subset of e4m3, so the weight side adds no error beyond the
requant. 27/27 parity green on the sm90 cell.

## Rule

On Hopper, fp8 WGMMA with on-the-fly fp4 dequant is dequant-bound: the e4m3
cast in the K-loop caps efficiency at ~20% of peak. A per-32 (not per-16)
weight scale is required to keep one scale per WGMMA K-tile and avoid the
temp-fragment epilogue. e4m3 multiplicative quant error does not average
down over K — gate fp8 kernels against an identical-quant reference, not f32.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | d51a5bc | H20 (contended) | sm90 fp8 | qwen36-27b slice2 (2 GDN) | 0.1276 | — | 7839 (slice) / 399 (extrapolated) |
| 2026-08-24 | 8ce52b4 | H20 (contended) | sm90 bf16 | qwen36-27b slice2 (2 GDN) | 0.1661 | — | 6021 (slice) / 268 (extrapolated) |

Raw artifacts: `/tmp/profile_fp8v2.out`, `/tmp/bench_fp83.out` (pod).
