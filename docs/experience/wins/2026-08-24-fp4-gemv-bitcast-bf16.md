# fp4 GEMV: bitcast fast decode + bf16 IO — decode 14% -> 27% of roof, slice 4.55 -> 3.91 ms/tok — H20, 2026-08-24

> Status: Shipped

## Context

The fp4 decode GEMV (`make_linear_fp4_gemv`, 733cbcd) ran at 12-16% of HBM
bandwidth on the big shapes. Two known gaps: (a) the scalar e2m1fn decode did
4 `exp2` per thread per K-step; (b) f32 IO. The tilelang lop3 intrin group
only covers affine uint/int grids — e2m1fn needs its own fast decode. Driver:
`scripts/bench_fp4_gemv.py /host/tc27-nvfp4-slice2 --layers 2` (H20, idle
GPU 7, JIT-free after same-shape warmup), same process before/after.

## What Worked

- **Bitcast fast decode.** The e2m1fn grid is a power-of-two grid, so each
  nibble's fp32 bit pattern is pure integer math — `sign<<31 | (126+e)<<23 |
  m<<22` — reinterpreted as float. No `exp2`, no LUT, no warp cooperation.
  Diagnostic sweep (`scripts/_sweep_gemv.py`, N=17408 K=5120) on the trade:

  | variant | ms | %roof |
  |---|---:|---:|
  | f32 exp2 decode (shipped) | 0.336 | 6.0 |
  | f32 nodecode (w=1, floor) | 0.073 | 27.7 |
  | bf16 exp2 decode | 0.330 | 6.1 |
  | bf16 local-array LUT | 0.192 | 10.5 |
  | bf16 warp-shuffle LUT | 0.125 | 16.2 |
  | bf16 bitcast (shipped) | 0.125 | 16.1 |
  | bf16 nodecode (floor) | 0.067 | 30.1 |

  Decode is ~60% of the f32 kernel (nodecode floor 0.073 vs 0.336). The
  local-array LUT spills to local memory and barely helps; the warp-shuffle
  LUT and the bitcast tie — the bitcast shipped (pure integer TIR, no
  warp-cooperation constraint, works for any `reduce_thread`).

- **bf16 IO.** X/Y bf16, f32 accumulate (`micro_size_k` 4 -> 8, block_K
  128 -> 256). The three MMA gemms (`gemm_nt/nn/tn_mma`) and
  `linear_fp4_mma` also converted: bf16 WGMMA (m16n8k16, 2x the TF32
  m16n8k8 throughput), f32 accumulate. The backend keeps bf16 from the
  boundary (the model is bf16-master) instead of casting bf16->f32->bf16.

- **Parity: bf16 IO holds at rtol=1e-2.** 25/25 CUDA parity tests green
  (`tests/test_ops_parity.py`, tiny shapes, vs torch-eager reference). The
  only error source is bf16 rounding of X and Y; the dequantized weight is
  exact (the e2m1fn LUT values are all exactly representable in bf16).

- **Per-linear GEMV, before (f32) vs after (bf16+bitcast):**

  | shape (N,K) | before ms | after ms | speedup | before %roof | after %roof |
  |---|---:|---:|---:|---:|---:|
  | 5120,17408 | 0.1529 | 0.0860 | 1.78x | 13.4 | 23.9 |
  | 17408,5120 | 0.1382 | 0.0774 | 1.79x | 14.9 | 26.5 |
  | 10240,5120 | 0.0849 | 0.0594 | 1.43x | 14.2 | 20.4 |
  | 248320,5120 (lm_head) | 1.8475 | 0.8949 | 2.06x | 15.9 | 32.7 |

  The WGMMA (M>1) path also gained: 17408,5120 0.3026 -> 0.1991 ms (1.52x,
  bf16 WGMMA). Small N shapes (48,5120) are launch-bound and unchanged
  within noise.

- **Slice decode (2 GDN layers):** GEMV path GPU 4.545 -> 3.905 ms/tick
  (1.16x), `linear_fp4` 3.608 -> 2.609 ms (1.38x, 79% -> 67% of the tick).
  vs the WGMMA-only baseline: 6.197 -> 3.905 ms/tick (1.59x).

## Rule

For a power-of-two quant grid (e2m1fn, e2m1, e3m2), the fast decode is
integer bit-pattern synthesis into the target float's IEEE fields — no LUT,
no `exp2`. bf16 IO is a free 1.5-2x on sm90 once the decode is cheap (it is
hidden by decode cost otherwise); parity at rtol=1e-2 holds because the
dequantized values are exactly bf16-representable.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | 733cbcd | H20 | cuda/sm90 | 27B slice (2 GDN layers) | — | 4.545 (GPU) | 187 decode |
| 2026-08-24 | this | H20 | cuda/sm90 | 27B slice (2 GDN layers) | — | 3.905 (GPU) | 213 decode |

Decode ms/tick is the profiler's per-tick GPU sum (10-tick average,
`scripts/profile_slice.py` path via `bench_fp4_gemv.py`); wall is 4.689
ms/tick (213 tok/s). The GEMV roofline gap is now the grid/reduction floor
(nodecode 30% of roof), not the decode — the next lever is the split-K
partition, not a faster decode.

Raw artifacts: pod `/work/bench_before.log`, `/work/bench_after2.log`,
`/work/sweep_gemv.log` (H20, GPU 7, JIT-free).
