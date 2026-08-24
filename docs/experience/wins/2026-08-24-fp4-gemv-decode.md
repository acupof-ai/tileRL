# fp4 GEMV decode kernel: slice decode 10.58 -> 5.45 ms/tick — H20, 2026-08-24

> Status: Shipped

## Context

`make_linear_fp4_gemv` (733cbcd) is the M=1 decode path of `linear_fp4` on
sm90: one warp group per 4 output rows streams WQ+Scale once, dequantizing
e2m1fn on the fly, instead of padding the activation to 16 WGMMA rows
(15/16 over-compute). It landed with CPU parity green but was never run on
CUDA — the previous verification workflow was stopped mid-flight. This entry
is the CUDA verdict: parity on GPU, per-linear GEMV vs WGMMA-padded, and
slice decode before/after in the same process. Driver:
`scripts/bench_fp4_gemv.py /host/tc27-nvfp4-slice2 --layers 2` (H20, idle
GPU 5, measured BW 3306.7 GB/s; 5 warmup + 50 timed iters per shape; the
"before" arm pops the GEMV key from the sm90 cell, so both arms go through
the same backend entry with identical Python overhead).

## What Worked

- **CUDA parity 23/23**, including `test_linear_fp4_gemv_parity`
  ((24,32), (16,128), (18,64), allclose rtol=1e-2 vs the torch-eager
  reference). The adaptation is a faithful SOTA copy
  (`examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py` @ tilelang
  main): f32 IO (micro_size_k = 128/32 = 4), e2m1fn grid decode in place of
  the example's uint4->int4 convert, tileRL's per-16 float block scale per
  micro-tile, M fixed at 1.
- **GEMV beats WGMMA-padded on every linear**, 1.3x–5.9x:

  | shape (N,K) | roof ms | GEMV ms | WGMMA ms | %roof | speedup |
  |---|---:|---:|---:|---:|---:|
  | 5120,17408 | 0.0202 | 0.1638 | 0.9588 | 12.4% | 5.9x |
  | 17408,5120 | 0.0202 | 0.1486 | 0.3262 | 13.6% | 2.2x |
  | 10240,5120 | 0.0119 | 0.0905 | 0.3063 | 13.1% | 3.4x |
  | 6144,5120 | 0.0071 | 0.0554 | 0.2693 | 12.9% | 4.9x |
  | 5120,6144 | 0.0071 | 0.0566 | 0.3260 | 12.6% | 5.8x |
  | 248320,5120 (lm_head) | 0.2884 | 1.9971 | 2.5629 | 14.4% | 1.3x |
  | 48,5120 (small) | 0.0001 | 0.045–0.058 | 0.216 | 0.1% | 3.7–4.8x |

- **Slice decode (2 GDN layers, avg of 10 ticks)**: GPU 9.937 -> 4.804
  ms/tick, wall 10.577 -> 5.452 ms/tick (94.5 -> 183.4 tok/s, 1.94x).
  `linear_fp4` itself: 8.905 -> 3.706 ms (2.4x), 90% -> 77% of the GPU
  tick. Consistent with the rebench entry's 9.949 ms/tick decode-only
  baseline (different idle GPU).
- **Roofline headroom**: 12–14% of measured HBM BW on the big shapes —
  memory-bound but far from peak. The scalar e2m1fn decode (4 `exp2` per
  thread per K-step) and f32 IO are the likely caps; the SOTA example's
  lop3 fast-decode path is the day-2 lever. Small shapes (48,5120) are
  launch-overhead bound (~50 us vs 0.1 us roof) — no schedule change
  helps there, fewer launches does.
- **Naive full-model extrapolation** (same caveats as the rebench entry):
  the ~188 ms linear_fp4 share of the ~239 ms/tick corrected 64-layer
  estimate scales by 3.706/8.905 -> ~78 ms, so ~129 ms/tok (~7.7 tok/s)
  vs the 80 tok/s target. GEMV is necessary but not sufficient; dispatch
  overhead and the GDN prefill reference still dominate the gap.

## Rule

At M=1 on sm90, a streamed-dequant GEMV strictly beats WGMMA padded to 16
rows (1.3–5.9x on every fp4 linear, 1.94x on slice decode) — never pad a
decode gemv to a tensor-core tile. Roofline headroom is real (12–14%) but
lives in the decode ALU (scalar exp2) and launch count, not in the weight
streaming.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | 733cbcd | H20 | cuda/sm90 | 27B slice (2 GDN layers) | — | 10.577 (WGMMA) / 5.452 (GEMV) | 94.5 / 183.4 decode |

Decode ms/tick is `profile_slice.time_decode` wall avg of 10 ticks (same
process, same GPU, GEMV key toggled between arms). Parity: 23/23 on CUDA
(GPU 5), 66 passed + 1 skipped on CPU.

Raw artifacts: pod `/tmp/bench_gemv.log` (H20, GPU 5, JIT-free after
warmup).
