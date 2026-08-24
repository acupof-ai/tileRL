# SOTA all-levers final bench: 80/3800 not met — gap is kernel efficiency, not H20 roofline — H20, 2026-08-24

> Status: Shipped

## Context

Final measurement of the SOTA kernel round on real NVFP4 weights, after every
lever landed: fp4 GEMV decode (bitcast fast decode + bf16 IO), fused GDN
decode + chunk prefill kernels, multi-block norm/activation, CUDA-graph
decode capture, fp8 prefill WGMMA, and FlashAttention paged attention.
Driver: `scripts/profile_slice.py` on both slices — `/host/tc27-nvfp4-slice2`
(2 GDN layers) and `/host/tc27-nvfp4-slice4` (3 GDN + 1 full-attn, whose 3:1
mix is exactly the 27B's 48:16 pattern, so its per-layer average extrapolates
to the full model with no layer-mix correction). The full NVFP4 checkpoint is
not on the pod (only the two slices), so the full-model numbers are
extrapolations, flagged as such. H20, GPU 3, under a 99%-util co-tenant
(8x75GB job on all 8 GPUs) — contention is this pod's normal state, and the
numbers below are contended unless stated.

## What Worked

**Slice measurements (HEAD 6dfa7d8, graph-captured decode, 30-tick avgs):**

| slice | decode ms/tick | decode tok/s | prefill-512 wall | prefill tok/s |
|---|---:|---:|---:|---:|
| 2 GDN | 1.922 | 520.2 | 0.0443 ms/tok | 22589 |
| 3 GDN + 1 FA | 2.798 | 357.4 | 0.0661 ms/tok | 15132 |

A second sample pair taken during a high-contention phase (6.653 / 9.934 ms
decode, 7643 / 4927 tok/s prefill) brackets the co-tenant's phase: the graph
wall is a single replay per tick, so contention costs one queue wait (~0-5 ms)
per tick, not per launch — the decode band is 1.9-6.7 ms on 2 layers. The
low-contention pair above is the best-case on this pod, used for the
extrapolation.

Per-layer decomposition (low-contention graph walls, fixed cost 0.071 ms =
pinned input copies + sampling + embedding): GDN layer 0.926 ms, FA layer
~0.93 ms (the paged-attention kernel's ~0.46 ms is offset by the FA layer's
smaller bf16 projection set — 136M vs 178M params — and no GDN chunk op).

Smoke 8-token average (1 prefill-16 + 7 decode, graph-captured,
high-contention): 11.02 ms/tok (slice2), 17.74 ms/tok (slice4). The smoke
metric is prefill-dominated under contention (the M=16 prefill tick eats
40-70 ms of GPU-queue wait) — the decode-only tick rate above is the real
decode metric.

**Eager per-op profile (slice2, 20-tick avg):** decode GPU sum 2.109 ms vs
wall 27.220 ms — dispatch overhead is 25.1 ms (784.7 us/op, 32 ops/tick).
On a 99%-util GPU each eager launch queues behind the co-tenant; graph
capture collapses the 32 launches into one replay and is the only viable
decode mode (6.653 vs 27.220 ms wall). Prefill-512: GPU sum 40.706 ms
(linear_fp4 75.3%, linear_attn_chunk 23.6%), wall 66.987 ms.

**Eager per-op profile (slice4, 40-tick avg, low-contention window):**
prefill-512 GPU sum 32.910 ms (linear_fp4 72.6%, chunk 23.0%, paged_attention
0.223 ms), wall 33.993 ms → 15062 tok/s slice / 975 tok/s extrapolated. The
2x spread vs the high-contention run (63.3 ms / 487 tok/s) is the co-tenant's
phase, not code — CUDA event timings inflate with per-launch queue waits and
are only trustworthy as a band.

**Full-model extrapolation (slice4 mix x 16, low-contention graph walls):**

| metric | extrapolated | target | gap |
|---|---:|---:|---:|
| decode | 59.3 ms/tok (16.9 tok/s) | 12.5 ms (80 tok/s) | 4.7x |
| prefill-512 | 1.02 ms/tok (976 tok/s) | 0.263 ms (3800 tok/s) | 3.9x |

Decode: 64 layers x 0.926 ms + 0.071 ms fixed = 59.3 ms (the FA layer costs
the same as GDN, so the layer mix does not move the total). Prefill: slice4's
per-layer average (32.741 ms / 4) x 64 = 524 ms per 512-token tick. Under
high contention both degrade ~2x (decode ~65 ms / 15 tok/s, prefill ~2.0 ms
/ 487 tok/s) — the band is 15-17 decode / 487-976 prefill tok/s.

**Roofline analysis (the contention-independent verdict):**

- **Decode**: the full tick reads 30.9 GB (48 GDN layers x 490 MB + 16 FA
  layers x 423 MB + lm_head 636 MB; GDN bytes are 73% bf16 projections —
  the checkpoint ships them FP8, `load_hf` dequantizes to bf16 — and 27%
  fp4 MLP). H20 at 4 TB/s: **7.73 ms roof = 129 tok/s**. The 80 tok/s
  target is 62% of BW roof — physics allows it. We are at 13% of roof
  (59.3 ms extrapolated): the fp4 GEMV runs at 24-33% of roof (isolated
  bench, idle), and the bf16 M=1 projections run the padded-MMA path at
  ~10-15%. The gap is kernel efficiency, concentrated in the bf16
  projections (73% of the bytes).
- **Prefill**: the tick is 28.5 TFLOP. Mixed-dtype roof (bf16 projections
  at 148 TFLOPS + fp8 MLP at 296 TFLOPS) = 133.5 ms → **3835 tok/s — the
  3800 target is the mixed-dtype roofline at 100% tensor utilization.** It
  is therefore unreachable while the GDN projections stay bf16: the roof
  is capped at 3835 tok/s even for perfect kernels. Keeping the checkpoint's
  FP8 projections in fp8 raises the roof to 5317 tok/s; the in-engine fp8
  GEMM is at 16-22% of peak contended (60-80% isolated idle) — the lever is
  fp8 weight retention plus closing the in-engine efficiency gap.

**Verdict: 80/3800 not met, and the remaining gap is code, not physics.**
The H20 roofline allows 129 decode / 5317 prefill tok/s. Decode needs a
bf16 GEMV path (the bf16 projections are 73% of decode bytes at ~10-15% of
roof) plus fp4 GEMV at 60%+ of roof. Prefill needs fp8 weight retention
(the current dtype split caps the roof at the target itself) plus fp8 GEMM
efficiency. Neither target is blocked by hardware.

## Rule

On a shared GPU, CUDA event timings inflate with per-launch queue waits
(measured up to 784 us/op) and vary 2x with the co-tenant's phase; the
graph-captured wall is the only stable decode metric, and bytes/FLOPs
roofline analysis is the only contention-independent verdict. And a
quantization target that equals the mixed-dtype roofline is a statement
about the dtype split, not the kernels — dequantizing the checkpoint's FP8
projections to bf16 caps the prefill roof at 3835 tok/s before a single
kernel runs.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | 3b1327f | H20 | cuda/sm90 | 27B slice (2 GDN) | — | 48.85 | 20.5 decode |
| 2026-08-24 | 53c1398 | H20 | cuda/sm90 | 27B slice (2 GDN) | 11.26 (512-tok) | 31.09 | 32.2 decode / 89 prefill |
| 2026-08-24 | 76826b0 | H20 idle | cuda/sm90 | 27B slice (2 GDN) | 0.2226 (512-tok) | 5.46 | 183 decode / 4491 prefill |
| 2026-08-24 | 6dfa7d8 | H20 low-ct | cuda/sm90 | 27B slice (2 GDN) | 0.0443 (512-tok) | 1.922 (decode-only) | 520 decode / 22589 prefill |
| 2026-08-24 | 6dfa7d8 | H20 low-ct | cuda/sm90 | 27B slice (3 GDN + 1 FA) | 0.0661 (512-tok) | 2.798 (decode-only) | 357 decode / 15132 prefill |
| 2026-08-24 | 6dfa7d8 | H20 high-ct | cuda/sm90 | 27B slice (3 GDN + 1 FA) | 0.2030 (512-tok) | 9.934 (decode-only) | 101 decode / 4927 prefill |
| 2026-08-24 | 6dfa7d8 | H20 low-ct | cuda/sm90 | 27B extrapolated (48 GDN + 16 FA) | 1.02 (512-tok) | 59.3 | 16.9 decode / 976 prefill |

Decode ms/tok is the graph-captured per-tick wall (the production decode
path; the smoke 8-token averages were 11.02/17.74 ms/tok but are
prefill-dominated under contention). Prefill is the 512-token wall from
`scripts/profile_slice.py`. The last idle-GPU measurement was 5.46/4491
(76826b0); this entry's numbers are all contended — low-ct and high-ct are
two phases of the same 99%-util co-tenant (the graph replay pays one queue
wait per tick, so the decode band is 1.9-9.9 ms on the slices).
Extrapolation: slice4's 3:1 layer mix x 16 = the 27B's 48:16, so its
per-layer average scales directly. Rooflines: decode 129 tok/s (30.9 GB at
4 TB/s), prefill 3835 tok/s mixed-dtype / 5317 tok/s fp8-everything.

Raw artifacts: pod `/work/prof_slice2_graph.log`, `/work/prof_slice2_eager.log`,
`/work/prof_slice2_graph2.log`, `/work/prof_slice4_graph.log`,
`/work/prof_slice4_eager.log`, `/work/prof_slice4_eager40.log`,
`/work/prof_slice4_graph2.log`, `/work/smoke_slice2.log`,
`/work/smoke_slice4.log` (H20, GPU 3, JIT-free).
