# GEMV + chunk-kernel round: slice decode 31.09 -> 5.46 ms/tok, prefill 89 -> 4491 tok/s — H20, 2026-08-24

> Status: Shipped

## Context

Final benchmark of the GEMV + chunk-kernel round on the 2-layer NVFP4 slice
(`/host/tc27-nvfp4-slice2`, both layers GDN), both kernels default-on in the
sm90 cell: `make_linear_fp4_gemv` (M=1 decode, 733cbcd) and
`make_gdn_chunk_fused` (T>1 prefill, 76826b0). Driver:
`scripts/profile_slice.py /host/tc27-nvfp4-slice2 --layers 2` (H20, idle
GPU 1, JIT-free after same-shape warmup) plus the smoke 8-token average
(`scripts/real_ckpt_smoke.py --gen 8 --train-steps 0`, same config as the
31.09 ms/tok baseline).

## What Worked

- **Smoke 8-token average: 48.85 -> 31.09 -> 5.46 ms/tok** (1 prefill [1,16]
  + 7 decode). Decode-only profile: wall 5.335 ms/tick (187.4 tok/s).
  Prefill 512: 0.2226 ms/tok (4491 tok/s) vs 11.26 ms/tok (89 tok/s) before
  the chunk kernel — the 3800 tok/s slice prefill target is met.
- **Decode per-op profile** (avg of 10 ticks): GPU sum 4.623 ms, wall
  5.335 ms, dispatch overhead 0.713 ms (22.3 us/op, 32 ops/tick).

  | op | ms | % |
  |---|---:|---:|
  | linear_fp4 | 3.513 | 76.0 |
  | rmsnorm | 0.440 | 9.5 |
  | linear_attn_chunk | 0.405 | 8.8 |
  | embedding | 0.116 | 2.5 |
  | silu_mul | 0.109 | 2.4 |
  | add | 0.023 | 0.5 |
  | sample | 0.015 | 0.3 |

- **Prefill 512 per-op profile** (one tick): GPU sum 113.344 ms, wall
  113.996 ms, dispatch overhead 0.651 ms.

  | op | ms | % |
  |---|---:|---:|
  | linear_fp4 | 62.814 | 55.4 |
  | silu_mul | 44.807 | 39.5 |
  | linear_attn_chunk | 5.033 | 4.4 |
  | rmsnorm | 0.501 | 0.4 |
  | embedding | 0.125 | 0.1 |
  | add | 0.048 | 0.0 |
  | sample | 0.015 | 0.0 |

- **Extrapolation to 64 layers.** The profiler's naive math (lm_head counted
  per layer) gives 165.26 ms/tok decode (6.1 tok/s) and 7.08 ms/tok prefill
  (141 tok/s). Corrected (lm_head is once-per-tick, ~2.0 ms from the GEMV
  bench): decode ~102 ms/tok (~9.8 tok/s), prefill ~6.95 ms/tok (144 tok/s).
  Targets: 12.5 ms/tok (80 tok/s) decode, 0.263 ms/tok (3800 tok/s) prefill
  — gap 8.2x decode, 26x prefill. Caveat: the slice has 2 GDN layers and 0
  full-attn layers; the 27B's 16 full-attn layers are unmeasured (GDN
  per-layer cost used as the average undercounts them).

## Rule

On sm90 the slice is no longer GEMM-bound: decode is 76% fp4 GEMV at 12-14%
of HBM BW behind 899 launches/tick, prefill is 55% M=512 GEMM plus 40% a
single-block silu_mul — the next round is launch count and elementwise
grids, not new GEMM schedules.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | 3b1327f | H20 | cuda/sm90 | 27B slice (2 GDN layers) | — | 48.85 | 20.5 decode |
| 2026-08-24 | 53c1398 | H20 | cuda/sm90 | 27B slice (2 GDN layers) | 11.26 (512-tok) | 31.09 | 32.2 decode / 89 prefill |
| 2026-08-24 | 76826b0 | H20 | cuda/sm90 | 27B slice (2 GDN layers) | 0.2226 (512-tok) | 5.46 | 183 decode / 4491 prefill |

Decode ms/tok is the smoke-bench 8-token average (1 prefill + 7 decode); the
decode-only profile rate is 187.4 tok/s (5.335 ms/tick). Prefill is the
512-token measurement from `scripts/profile_slice.py`. Extrapolated full
model (lm_head corrected): ~102 ms/tok decode (9.8 tok/s) and 6.95 ms/tok
prefill (144 tok/s) vs 80/3800 targets.

What the profile says is next:

- **Decode (8.2x off)**: per-layer fp4 gemms are ~78% of the extrapolated
  tick — GEMV roofline headroom (12-14% of BW) lives in the scalar e2m1fn
  decode and f32 IO (lop3 fast-decode path, bf16 IO). Dispatch overhead is
  22.3 us/op x 899 ops/tick = 20 ms — 1.6x the entire 12.5 ms target by
  itself; fewer, fused launches are mandatory. rmsnorm is 9.5% (5
  single-block calls/tick, launch-bound).
- **Prefill (26x off)**: `silu_mul` is a single-block kernel
  (`T.Kernel(1)`, 64 threads) over 512x17408 = 8.9M elements — 44.8 ms of
  the 113 ms tick. A multi-block grid takes it to memory-bound (~0.03 ms
  roof); that is the single biggest prefill lever. Then the M=512 fp4
  gemms (55%).

Raw artifacts: pod `/work/profile_final.log`, `/work/smoke_final.log`
(H20, GPU 1, JIT-free).
