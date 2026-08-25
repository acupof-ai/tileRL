# Chunked GDR scan for GDN prefill — REJECTED (0.90x, bf16 precision wall), 2026-08-25

## Context

The serial GDN prefill mega-kernel (`make_gdn_chunk_fused`, one launch,
serial-within-block scan over T tokens) is 27.4% of the prefill-512 tick.
The FlashQLA chunk-wise gated-delta-rule pipeline (6 kernels: cumsum, kkt,
solve, recompute, state, o) was ported from agent-infer as a replacement —
the block reordering of the serial scan, exact for an affine-in-S0
recurrence. Wired as the default for both the 6-arg scan and the full-GDN
prefill path (conv1d+SiLU → norm+gates → chunked scan → post-norm+z) on
sm90, K=V=128, T>1.

A/B at slice4 prefill-512 shapes (B=1, T=512, nkh=16, nvh=48, K=V=128),
same-process, same random inputs, quiet H20 window:

| arm | GPU ms (median, 20 iters) | vs reference |
|---|---:|---|
| serial mega-kernel | 4.380 | exact (max\|d\|=0.0000) |
| chunked-WY pipeline | 4.882 | **max\|d\|=8.51 out, 1.37 state** |

The chunked pipeline is 10% slower **and** wrong at realistic input scale.

## Root Cause

**Performance.** The WY formulation is O(T·C·K) per head (C=64: the A
construction, u/w matmuls, and output are all C²-contractions), while the
serial scan is O(T·K²). The serial kernel is compute-bound, not
memory-bound: its 64KB state tile per head (48 heads = 3MB) fits in L2, so
the "state streamed from HBM" is actually L2 traffic. The 48 value heads
already saturate the SMs (78 on H20), so the chunked pipeline's 8× more
blocks don't add useful parallelism. This matches the previous WY rejection
(`2026-08-25-gdn-prefill-wy-rejected.md`, 2.6x slower with a 2-kernel port —
the FlashQLA 6-kernel pipeline closes most of that gap but can't overcome
the fundamental algorithmic disadvantage).

**Precision.** The chunked pipeline stores intermediates (a_inv, w, u,
v_new) in bf16 global memory between stages. At scale=0.1 inputs (test
shapes) the error is 2% (passes rtol=1e-2). At scale=1.0 (the bench's
input distribution, matching the model's post-conv1d SiLU magnitude) the
error is 26% — the bf16 rounding on the ~1.5-valued v/u/v_new intermediates
accumulates over the 6-stage pipeline. The serial mega-kernel fuses all
stages and keeps intermediates in f32/registers, so it is exact (0.0000
error) at the same scale. Bisect: scale=0.1 passes (0.8% error), scale=1.0
fails (26% error); g distribution and T length are not triggers.

**Bug found and fixed (kept).** `gdr_solve` writes only the 10
lower+diagonal blocks of the 64×64 inverse; the 6 upper off-diagonal blocks
are zero (inverse of lower-triangular is lower-triangular) but never
written. A fresh process hides this (CUDA memory is zeroed), but a reused
buffer leaks stale values into recompute's GEMM — NaN on the second call.
Fix: `a_inv.zero_()` in the backend before launch. This is a genuine bug
fix that stays regardless of the rejection.

## Fix

Reverted the chunked pipeline as the default for both paths:

- **Full-GDN prefill**: removed the staging kernels (conv1d_silu,
  norm_gates, post_norm_z), `_gdn_chunk_prefill`, and the wiring in
  `linear_attn_chunk`. The serial mega-kernel is the default for T>1.
- **6-arg scan**: removed the wiring in `linear_attn_chunk`. The portable
  serial kernel is the default for all shapes/arches.

Kept:

- The 6 GDR kernels (`make_gdr_*`) and `_gdr_chunk_scan` — verified SOTA
  code (parity green at scale=0.1), registered in the sm90 kernel cell,
  available for future shapes/models where the serial kernel is
  memory-bound or parallelism-starved (few heads, state > L2).
- `test_gdr_chunk_scan_parity` — calls `_gdr_chunk_scan` directly, keeping
  the kernels verified.
- The `a_inv.zero_()` fix, the `test_gdn_chunk_matches_decode` SeqQLens
  fix, and the fla chunk-delta-rule docstring correction.

## Rule

Don't replace a serial scan with a chunkwise WY scan when the state fits in
L2 and heads ≥ SMs/2 — the O(T·C) extra FMAs cost more than the
state-traffic savings, and bf16 intermediates between pipeline stages lose
precision at realistic input scales. WY pays off only when the serial
kernel is memory-bound (state > L2) or parallelism-starved (few heads),
AND the intermediates can stay in f32. Measure both; the 27.4% tick share
was state-L2 traffic + serial latency, not HBM.
