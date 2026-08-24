# fp4 prefill vectorized dequant — slice4 15293→17970 tok/s, gate/up 180 TFLOPS (61% peak) — H20, 2026-08-25

> Status: Shipped

## Context

The fp4 prefill path (`linear_fp4_fp8` fp8-IO, `linear_fp4` bf16-IO) dequantized
each e2m1fn nibble serially in the K-loop: a per-element `T.Parallel` decode
(integer bitcast + per-element global scale load + e4m3 requant) between the
WQ copy and the WGMMA. On the MLP projections (NVFP4, no native fp8 checkpoint
weights to preserve — dequant-in-loop is unavoidable) this held the kernel at
107-125 TFLOPS (36-42% of the 296 TFLOPS fp8 peak) and made `linear_fp4` 62% of
the slice4 prefill tick. The Hopper dequant_gemm SOTA
(`example_dequant_gemm_bf16_fp4_hopper.py`, fast_dequant path) vectorizes the
dequant into shared memory ahead of the WGMMA, pipelined so the dequant hides
behind the MMA. This entry ports that schedule.

## What Worked

**Vectorized shared-memory dequant macro (`_dequant_fp4_macro`).** Each K-loop
stage copies the X/WQ/Scale tiles to shared, then a `T.Parallel` chunk loop
dequants the WQ tile into W_shared: 128-bit transactions (16 e4m3 elems out /
8 packed bytes in per chunk), the e2m1fn integer bitcast decode (no extern —
the SOTA's twiddling intrin only covers affine int4 grids), one per-32-block
scale per chunk (staged to shared). The WGMMA then reads W_shared. With
`num_stages=3` the dequant of stage k+1 issues while the WGMMA of stage k is in
flight (async WGMMA + double-buffered shared).

**T.Parallel, not T.serial, for the chunk loop.** A serial chunk loop
obstructs the K-loop software pipeliner on long K-loops — the dequant can't
overlap the WGMMA — costing 0.85-0.89x on K=17408 (down_proj). T.Parallel lets
the pipeliner reorder the dequant past the WGMMA wait. Same-process old-vs-new
(`scripts/diag_dequant.py`, H20, M=512):

| shape (K,N) | old TFLOPS | new TFLOPS | speedup | % of 296 peak |
|---|---:|---:|---:|---:|
| 5120, 17408 (gate/up) | 124.4 | 180.0 | 1.45x | 61% |
| 17408, 5120 (down) | 108.6 | 115.3 | 1.06x | 39% |
| 5120, 10240 (qkv) | 107.0 | 121.5 | 1.14x | 41% |
| 6144, 5120 (out) | 106.8 | 115.9 | 1.09x | 39% |

The output is bit-identical to the old kernel (maxdiff 0.0 — the dequant math
is unchanged, only the schedule).

**Scale staged to shared, not read from global per chunk.** Reading the
per-32 scale from global (L2-cached) once per chunk loses to staging: 1.14x vs
1.24x on gate/up. The staging costs a 3rd `T.copy` per K-iteration (1 KB) but
removes the L2 latency from the dequant critical path.

**block_K 32→64 on the bf16 path** (was `_RED_TILE=32`): amortizes the
dequant over 4 WGMMA steps (bf16 K=16) instead of 2. The backend pads K to 64.

**Slice4 prefill (3 GDN + 1 FA, prefill-512, idle GPU):**

| arm | slice ms/tok | slice tok/s | linear_fp4 % of tick | extrapolated full-model tok/s |
|---|---:|---:|---:|---:|
| before (per-element dequant) | 0.0654 | 15293 | 62.3% | 992 |
| after (vectorized dequant) | 0.0556 | 17970 | 56.4% | 1172 |

1.18x slice, 1.31x on the `linear_fp4` op (20.9 → 16.0 ms). Per-op after:
linear_fp4 56.4%, linear_attn_chunk 27.6%, linear_fp8 10.7%, rmsnorm 2.8%,
paged_attention 0.8%.

**Rejected:**
- T.serial chunk loop: 0.85-0.89x on long-K shapes (obstructs the pipeliner).
- Global per-chunk scale (no staging): 1.08-1.14x vs 1.14-1.24x for staged.
- alloc_local inside vs outside the chunk loop: neutral (the compiler already
  handles liveness).

**3800 tok/s not met.** The extrapolated full-model prefill is 1172 tok/s
(20% of the 5898 tok/s roof) vs the 3800 target (64% of roof). The fp4 path
is now at 39-61% of peak on the MLP shapes; the remaining gap is the GDN chunk
op (27.6% of the tick, unchanged — a separate piece of work) and the linear
ops' peak efficiency (the native fp8 path tops out at ~50% of peak at these
tile sizes).

## Rule

For a dequant-in-the-mainloop GEMM on Hopper: (1) vectorize the dequant into
shared memory ahead of the WGMMA (128-bit transactions, one scale per chunk)
and let the K-loop pipeline hide it behind the MMA; (2) the chunk loop must be
`T.Parallel`, not `T.serial` — a serial loop obstructs the software pipeliner
on long K-loops, costing ~20%; (3) stage the per-block scale to shared (a 3rd
`T.copy` is cheaper than an L2 load on the dequant critical path). The
dequant-in-loop ceiling is ~60% of fp8 peak at 128×128 tiles — the rest is
tile size and the non-GEMM ops.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | c97f79c | H20 idle | cuda/sm90 | 27B slice (3 GDN + 1 FA) | 0.0654 (512-tok) | 2.662 (graph) | 15293 prefill / 376 decode |
| 2026-08-25 | 2672be6 | H20 idle | cuda/sm90 | 27B slice (3 GDN + 1 FA) | 0.0556 (512-tok) | — | 17970 prefill |
| 2026-08-25 | 2672be6 | H20 idle | cuda/sm90 | 27B extrapolated (48 GDN + 16 FA) | 0.853 (512-tok) | — | 1172 prefill |

Prefill is the 512-token wall (slice) / naive per-layer×64 extrapolation
(full-model, same method both rows). The before row is the final bench
(2026-08-25-bf16-gemv-fp8-weights.md, same slice, idle GPU). Per-shape TFLOPS
are same-process old-vs-new (`scripts/diag_dequant.py`). 31/31 CUDA parity
(`tests/test_ops_parity.py`, rtol=1e-2), 29/29 CPU.

Raw artifacts: pod `/work/diag_final.log` (old-vs-new per-shape),
`/work/bench_prefill_after.log` (bf16 vs fp8 path per shape),
`/work/profile_slice4_after.log` (slice4 per-op breakdown),
`/work/parity_full2.log` (31 CUDA parity).
