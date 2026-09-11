# 27B V100 prefill is 74-86% dense attention, not the GEMMs — sm70, 2026-09-11

> Status: **measured on the V100 sm70, f32 pool, 27B NVFP4**
> (`scripts/bench_prefill_split_v100.py`). The 25-35x prefill gap is dense
> attention; the linears are a flat 4.3 ms/token. This reorders the speed work
> ahead of the f16 tensor-core GEMM.

## Context

256k prefill on the V100 measured ~2.25 h (31 ms/token). Roofline said the
f16 prefill GEMMs alone should take 4-6 min, so ckl set two suspects: the sm70
prefill GEMM running without f16 tensor cores, and dense long-context
attention. This splits one real prefill tick into the three buckets.

## Method

One request prefills to ctx with CUDA-event pairs around every
`paged_attention(_prefill)` call (the 16 full-attn layers) and every
`linear_attn_chunk` (the 48 GDN layers); events are recorded without
per-call sync and read after one final sync. forward wall minus those two is
linears + norms + writes + lm_head ("other"). Warm + one measured draw per
ctx.

## Results

| ctx | forward s | attention s | GDN s | other/linear s | ms/token | attn share |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 593 | **441** | 10.5 | 142 | 18.11 | 74% |
| 65536 | 2144 | **1842** | 20.8 | 281 | 32.71 | 86% |

Dense attention grows linearly with history (13.5 ms per extra 1k tokens:
441/32.8 → 1842/65.5) and is three quarters of a 32k prefill, rising to 86%
at 64k. The linear bucket is flat at **4.3 ms/token** at both contexts —
weight-bound, context-independent, exactly the cost the serving pool's fixed
weight stream should have. GDN is 1-2%.

## What this changes

The f16 tensor-core prefill GEMM attacks the 4.3 ms/token linear bucket
(~18 min at 256k). Dense sparse-prefill attention over the selected ~136
pages (unit F's chunked prefill) attacks the bucket that already dominates
and is the only one whose share grows: at 256k dense attention extrapolates
to ~73 min vs the GEMMs' ~18. The decode-time measurement agrees: sparse
attention was 0.37 ms flat vs 3.46 ms dense at 64k per call (9.3x),
[2026-09-11-sparse-kv-v100-dense-vs-sparse.md](2026-09-11-sparse-kv-v100-dense-vs-sparse.md).

## Addendum — f16 M-tiled block GEMM halves the linear bucket

The prefill linear path used the M=32 GEMV ladder, which re-reads the whole
weight stream once per 32-row chunk. A block GEMM
(`linear_fp4_f16_mma_sm70`) dequantizes each fp4 weight tile to f16 in shared
once and sweeps a 64-row M tile through m8n8k4. Measured on the largest layer
(N=17408, K=5120, 27B NVFP4), CUDA-event median:

| launch M | ms / token-row |
|---:|---:|
| 32 (old ladder top rung) | 0.0093 |
| 64 | 0.0059 |
| 128 | 0.0058 |
| 256 (new block GEMM) | **0.0048** (1.94x) |

Parity vs the natural-pack f32 reference: 1.7-1.8e-3 end-to-end through
`Backend.linear_fp4` at M=64/256, and 3.2e-4/1.9e-4 on the standalone kernel
(M=32..256). M<=8 keeps the GEMV ladder (a single vector is bandwidth-bound).

Debugging note: an apparent rel 1.69 through the wrapper was a probe bug, not a
kernel bug — `tensor._tl_layout` was set before `.cuda()`, and a device move
strips the custom attribute, so `_served_fp4` saw "natural" and TWIDDLED THE
BYTWIDDLED BYTES (double twiddle). Tagging after `.cuda()` (as `materialize`
does in production) gives rel 0.0018. Rule: a `_tl_layout` tag is attached to
the GPU tensor after migration, never before a device move.

## Rule

Before rebuilding a kernel for a roofline gap, time the tick split per
phase: a 25-35x end-to-end gap read as "the GEMM is on SIMT" can be a
different phase entirely. Here the GEMMs were already on f16 MMA (tw-f16
M-ladder) and flat per token; the quadratic attention phase carried the gap.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-11 | V100-SXM2-32GB | cuda sm70 | prefill attn 74%@32k / 86%@64k; linear 4.3 ms/tok flat; 593 s / 2144 s total |
| 2026-09-12 | V100-SXM2-32GB | cuda sm70 | f16 block GEMM 0.0093→0.0048 ms/row at M256 (1.94x); end-to-end rel 1.8e-3 |
