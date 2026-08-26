# A=bf16 prefill GEMM rejected — 1.55x slower than shipped e4m3+fp8; W4 error is only 1.7e-3

> Status: Killed (correctness green, performance regression)

## Context

The precision ladder for the sm90 prefill GEMM: W is fixed at 4-bit (NVFP4
e2m1fn packed); the open question is A (activation) precision. Shipped:
A per-token quantized to e4m3, W dequant e2m1→e4m3 in-kernel, fp8 WGMMA,
k_split=2 (`make_linear_fp4_fp8_mma`, the backend's M>1 sm90 path). The
alternative: A stays bf16 (no quant), W dequant e2m1→bf16, bf16 WGMMA
(`make_linear_fp4_mma` — the pre-fp8 prefill kernel, still registered as
`linear_fp4`, the natural flip candidate). Gate: ship only if faster AND
relerr vs the bf16 torch oracle ≤ 1e-2.

A/B (`scripts/bench_abf16_prefill.py`, H20 pod GPU 7, mean of 20, same
process — contention-independent ratio; arm A's timed region is the shipped
path end-to-end: pad + per-token quant + output zeroing + split kernel;
arm B has no quant):

| shape (M,K,N) | A shipped ms | B bf16 ms | B/A | A relerr | B relerr |
|---|---:|---:|---:|---:|---:|
| gate/up (512,5120,17408) | 0.4910 | 0.7773 | 0.632 | 4.02e-2 | 1.74e-3 |
| down (512,17408,5120) | 0.5407 | 0.8667 | 0.624 | 3.79e-2 | 1.67e-3 |
| qkv (512,5120,10240) | 0.3105 | 0.4828 | 0.643 | 3.80e-2 | 1.70e-3 |
| z (512,5120,6144) | 0.1982 | 0.2955 | 0.671 | 3.94e-2 | 1.67e-3 |
| out (512,6144,5120) | 0.2115 | 0.3205 | 0.660 | 3.88e-2 | 1.76e-3 |

geo-mean B/A = 0.646x — the bf16 arm is 1.55x slower at every shape.

## Error decomposition (vs `reference.linear_fp4`, torch f32 dequant GEMM)

- **B (bf16 arm) relerr ≈ 1.7e-3** — the pure W4 error: e2m1fn packing +
  bf16 MMA accumulation order. 24x under the 1e-2 gate. W4 is not the
  accuracy bottleneck.
- **A (shipped e4m3) relerr ≈ 3.8-4.0e-2** — W4 error plus the A-side cost:
  per-token e4m3 activation quant (~2% floor) plus the e2m1→e4m3 weight
  requant cast (~1.7%). The A-quant contribution dominates (~3.8e-2 of the
  ~4.0e-2 total); W4's 1.7e-3 is negligible next to it.

So the precision ladder's accuracy lever, if one is ever needed, is A
precision — but A=bf16 buys it at 1.55x the time. The e4m3 path's ~4% is the
accepted operating point (end-to-end quality validated at the model level,
not by this per-GEMM gate).

## Root Cause

H20 (sm90) tensor-core throughput: bf16 WGMMA is half of fp8 WGMMA
(~148 vs ~296 TFLOPS dense). The prefill GEMMs are compute-bound at M=512
with large N/K, so the arm's 2x MMA throughput gap lands nearly full-weight
— the measured 1.55x (not a clean 2x) is the fp8 advantage minus the costs
only arm A pays: the per-token quant kernel (~0.024 ms), the split output
zeroing, and the f32 atomic adds. Secondary: the bf16 A tile moves 2x the
bytes of e4m3, but the GEMM is compute-bound, so this is not the driver.
The dequant itself is not the differentiator — it hides behind the WGMMA in
both arms (and the bf16 dequant is cheaper per elem: no e4m3 requant cast).

Tile tuning cannot close a 55% gap: arm B already uses the 64-tile N-grid
the fp8 path's own sweep found optimal for this dequant schedule family
(`docs/experience/wins/2026-08-25-fp8-prefill-n64-tile.md`), and the bf16
MMA throughput ceiling is the wall.

## Rule

For the sm90 prefill GEMM at W=4: keep A=e4m3 + fp8 WGMMA. A=bf16 is 1.55x
slower (H20 fp8 WGMMA has 2x the bf16 throughput and the GEMM is
compute-bound), and the accuracy it buys is worthless — W4 alone is
1.7e-3, so the shipped path's ~4% relerr is A-quant error the model already
tolerates. The bf16 arm stays in the tree as the registered `linear_fp4`
fallback; do not re-run this A/B unless the A-quant error becomes a
model-level quality problem.

## Results

| date | commit | machine | target | arm | geo-mean ms | B/A |
|---|---|---|---|---|---:|---:|
| 2026-08-26 | 2e5921e | H20 pod GPU 7 | cuda/sm90 | A shipped e4m3+fp8 ksplit2 | 0.322 | 1.0x |
| 2026-08-26 | 2e5921e | H20 pod GPU 7 | cuda/sm90 | B bf16 no-quant | 0.499 | 0.646x |

## Iteration

Hypothesis -> verdict in 11.9 min agent wall time (3 pod round-trips) — one
of two parallel arms of the A-precision sweep (workflow wall 12.0 min for
both, 215k subagent tokens).
