# Multi-block norm/activation: prefill silu_mul 46.1 -> 0.07 ms — H20, 2026-08-24

> Status: Shipped

## Context

The GEMV + chunk-kernel final bench (2026-08-24-gemv-chunk-kernels.md) flagged
two launch/parallelism-bound kernels in the portable floor: `silu_mul` was a
single `T.Kernel(1)` block with 64 threads over 512x17408 = 8.9M elements
(44.8 ms, 40% of the prefill tick), and `rmsnorm` was 5 single-block calls
per decode tick (9.5% of decode). Both are elementwise/grid work, not
compute — the fix is grids, not new schedules. Driver:
`scripts/profile_slice.py /host/tc27-nvfp4-slice2 --layers 2` (H20, idle
GPU 5, JIT-free after same-shape warmup), same config before and after.

## What Worked

- **`silu_mul`: grid over M in 1024-element chunks** (`T.ceildiv(M,
  block_M)` blocks, `T.if` tail guard). Prefill 512: **46.089 -> 0.070 ms**
  (658x, 38.5% -> 0.1% of the tick). Roof is ~0.03 ms (107 MB at 4 TB/s);
  the 2.3x gap is launch overhead plus the f32 sigmoid. Decode: 0.111 ->
  0.017 ms.
- **`rmsnorm`: two-kernel split-K.** Phase 1 (`make_rmsnorm_partial`)
  writes per-chunk sums of squares — grid over (chunks, rows), block_N=256,
  serial fragment-scalar accumulator per chunk. Phase 2
  (`make_rmsnorm_apply`) redundantly reduces the few chunk sums (cheap) and
  normalizes its own chunk, so both passes are parallel even at M=1.
  Decode: 0.445 -> 0.410 ms per tick (9.1% -> 8.5%); prefill: 0.520 ->
  0.403 ms. The win is small because the old kernel's serial loop was only
  ~50 us of the ~89 us per-call cost — the rest is per-launch dispatch, and
  split-K spends a second launch. Decode rmsnorm is now launch-bound at 2
  launches/call; the next lever is fusion, not more parallelism.
- **Portable source, no arch fork.** The tilelang example idiom
  (`examples/norm/rms_norm.py`, `T.reduce_sum` over a whole-row fragment) is
  not Metal-portable — Metal does not cross-thread-reduce fragments — so the
  per-chunk accumulator keeps the repo's serial fragment-scalar idiom, and
  the reduction across chunks is a second kernel instead of a grid sync.
  `T.if` tail guards lower on CPU (serial), CUDA, and Metal. CPU parity
  green; CUDA parity 25/25 (tiny shapes, allclose rtol=1e-2).

Prefill 512 per-op profile (one tick), before -> after:

| op | before ms | after ms |
|---|---:|---:|
| linear_fp4 | 67.479 | 67.599 |
| silu_mul | 46.089 | 0.070 |
| linear_attn_chunk | 5.338 | 5.365 |
| rmsnorm | 0.520 | 0.403 |
| GPU sum | 119.617 | 73.653 |
| ms/tok | 0.2336 | 0.1439 |
| tok/s | 4257 | 6886 |

Decode per tick (avg of 10), before -> after:

| op | before ms | after ms |
|---|---:|---:|
| linear_fp4 | 3.740 | 3.830 |
| rmsnorm | 0.445 | 0.410 |
| linear_attn_chunk | 0.399 | 0.422 |
| silu_mul | 0.111 | 0.017 |
| GPU sum | 4.861 | 4.835 |

## Rule

Elementwise kernels grid over their element range (one block per chunk);
row reductions split-K with a per-chunk workspace and a second apply kernel
— never a single serial block, and never the example `T.reduce_sum` idiom
on the portable floor (Metal does not cross-thread-reduce fragments). On
sm90 the slice prefill is now 92% fp4 GEMM; decode rmsnorm is launch-bound,
so the next norm/activation lever is fewer launches (fusion), not more
blocks.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick (GPU sum) | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | ae08a49 | H20 | cuda/sm90 | 27B slice (2 GDN layers) | 0.2336 (512-tok) | 4.861 | 4257 prefill |
| 2026-08-24 | f4618cc | H20 | cuda/sm90 | 27B slice (2 GDN layers) | 0.1439 (512-tok) | 4.835 | 6886 prefill |

Decode is the profile GPU-sum per tick (avg of 10), not the smoke 8-token
average. Extrapolated full-model prefill (profile's naive math): 7.47 ->
4.59 ms/tok (134 -> 218 tok/s) vs the 3800 tok/s target — the remaining
gap is the M=512 fp4 GEMMs (92% of the tick) and the 16 unmeasured
full-attn layers.

Raw artifacts: pod `/work/profile_before.log`, `/work/profile_after.log`,
`/work/parity_after.log` (H20, GPU 5, JIT-free).
