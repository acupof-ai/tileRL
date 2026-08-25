# fp8 prefill GEMM: N-tile 128 -> 64, +33% geo-mean TFLOP/s — H20, 2026-08-25

> Status: Shipped

## Context

`make_linear_fp4_fp8_mma` (fp4 weight -> e4m3 dequant + fp8 WGMMA, the M>1
prefill path of `linear_fp4`) ran at 59% of the 296 TFLOP/s fp8 peak on the
best shape (gate/up) and went NEUTRAL vs the bf16 fallback at large K
(down 1.03x, out 1.01x). The speedup vs bf16 was monotonic in N: 1.46x @
N=17408 (544 grid blocks, 2.3 waves) down to 1.01x @ N=5120 (160 blocks,
0.68 waves). Same FLOPs, same dequant work per FLOP — the difference was
wave count. Warp specialization is off (the SOTA dequant-gemm example
disables it too), so the dequant and WGMMA share threads; with under 1 wave
the resident blocks align in their dequant phase and the tensor cores idle.

Driver: `scripts/bench_fp8_prefill.py` (H20, idle GPU 0, JIT-warm), same
process before/after. Sweep: `scripts/_sweep_fp8_prefill.py` (6 variants,
5 shapes, parity vs the shipped kernel).

## What Worked

- **N-tile 128 -> 64 (`_FP4_BLOCK_N`).** Doubles the N-tile count, putting
  every prefill shape at 2+ waves so the dequant/WGMMA phases decorrelate
  across resident blocks. One constant; the dequant schedule, K-tile, and
  math are unchanged (exact parity, rel-base 0.00 vs the 128-tile kernel).

  Sweep (idle H20, TFLOP/s, `scripts/_sweep_fp8_prefill.py`):

  | shape (M,K,N) | baseline (128) | v_n64 | v_split2 | v_n64_split2 |
  |---|---:|---:|---:|---:|
  | 512,5120,17408 (gate/up) | 179.8 | 214.8 | 189.6 | 214.5 |
  | 512,17408,5120 (down) | 114.6 | 162.9 | 144.7 | 194.7 |
  | 512,5120,10240 (qkv) | 146.4 | 189.8 | 163.6 | 204.0 |
  | 512,5120,6144 (z) | 134.1 | 184.5 | 158.0 | 200.5 |
  | 512,6144,5120 (out) | 113.3 | 158.0 | 140.7 | 185.3 |

  v_n64 geo-mean +33%. Rejected variants: `v_int32` (block_K=32, integer
  e2m1->e4m3 dequant with the scale on the accumulator — 2x slower: the
  2x K-loop iterations and second accumulator fragment outweigh the cheaper
  dequant), `v_ws` (warp specialization on — slower, confirms the SOTA's
  disable), `v_sota` (256x128 tile, 1 block/SM — slower), `v_m64` (block_M=64
  — neutral, the dequant amortization loss offsets the wave gain).
  `v_n64_split2` (block_N=64 + 2-way K-split, atomic add) is another +8%
  geo-mean but needs a zeroed-output wrapper; not landed (follow-up).

- **Shipped kernel, via `backend.linear_fp4` (includes the 0.024ms quant):**

  | shape (M,K,N) | bf16 TFLOP/s | fp8 before | fp8 after | speedup vs bf16 |
  |---|---:|---:|---:|---:|
  | 512,5120,17408 (gate/up) | 123.5 | 175.8 | 209.5 | 1.70x (was 1.46x) |
  | 512,17408,5120 (down) | 111.0 | 113.8 | 157.5 | 1.42x (was 1.03x) |
  | 512,5120,10240 (qkv) | 117.7 | 143.4 | 183.4 | 1.56x (was 1.22x) |
  | 512,5120,6144 (z) | 117.8 | 127.3 | 175.0 | 1.49x (was 1.08x) |
  | 512,6144,5120 (out) | 107.6 | 103.8 | 150.5 | 1.40x (was 1.01x) |

  The down/out shapes that were neutral vs bf16 are now 1.40-1.42x.

- **Parity.** The tile change is math-neutral (rel-base 0.00 vs the 128-tile
  kernel in the sweep). Backend-level check vs `reference.linear_fp4`:
  rel-err 3.3-4.0% across the 5 shapes — the fp8 quantization floor
  (~2% activation e4m3 + ~1.7% weight requant), unchanged from the 128-tile
  kernel. `uv run pytest -q`: 72 passed, 4 skipped (GPU/Metal).

## Rule

For the fp4->fp8 dequant GEMM on sm90, the N-tile must be small enough to
keep 2+ waves on every target shape: 128 left the small-N grids (N=5120,
40 N-tiles) under 1 wave, so the dequant and WGMMA phases aligned across
resident blocks and the tensor cores idled. 64 doubles the grid for +33%
geo-mean with no math change. The wave count — not the K-loop length — is
the first-order lever for this kernel's occupancy.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | prev | H20 idle | cuda/sm90 | GEMM micro-bench (5 prefill shapes) | — | — | 175.8/113.8/143.4/127.3/103.8 TFLOP/s |
| 2026-08-25 | this | H20 idle | cuda/sm90 | GEMM micro-bench (5 prefill shapes) | — | — | 209.5/157.5/183.4/175.0/150.5 TFLOP/s |

TFLOP/s columns are gate/up, down, qkv, z, out (fp8 path, includes the
0.024ms per-token quant). Raw artifacts: pod `/work/sweep_fp8.log`
(sweep, 6 variants), `scripts/bench_fp8_prefill.py` output (shipped kernel).
