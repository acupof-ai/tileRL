# Changelog

Central progress record. Three event classes land a line the same day, linking
the `docs/experience/` entry: **phase exit · default flip · accept-or-reject
verdict**. Newest first.

## 2026-08-25 — accept-or-reject: dual-format fp8 weights for prefill — REJECTED (fp8 MMA already wired)

- **Verdict.** Proposed packing an e4m3 copy of every fp4 projection so
  prefill could use fp8 tensor cores. Killed by one bench: `backend.py:431`
  already routes `linear_fp4` through `make_linear_fp4_fp8_mma` on CUDA M>1 —
  prefill already computes on fp8 (176 TFLOP/s on gate/up, 1.46x over the
  bf16 fallback). The prefill gap is kernel efficiency, not format: the fp8
  MMA sits at 59% of peak and goes neutral at large K (down/out ~1.03x).
  Entry: `docs/experience/errors/2026-08-25-dual-format-fp8-rejected.md`.

## 2026-08-25 — default flip: serving fuses same-input fp4 projections (qkv/ab/gate_up) — decode +4.8%

- **Flip.** `cmd_serve` now loads with `fuse_projections=True`: same-input
  fp4 projections concat losslessly along N at load (per-32-block scales are
  per-row) and one GEMV replaces the group's launches. Training keeps the
  unfused masters (the fused key has none — its tape backward would have
  nowhere to land the STE grad). A/B on slice4, decode graph: 1.821 → 1.734
  ms/tick (549 → 577 tok/s slice); prefill +1.3% (noise). The win is
  per-kernel replay overhead, not Python dispatch (the graph already
  amortizes that) — the eager B>1 path stands to gain more. Entry:
  `docs/experience/wins/2026-08-25-projection-fusion-decode.md`.

## 2026-08-25 — accept-or-reject: fp4 GEMV shared-memory dequant ping-pong — REJECTED (2.5-3.3x slower than register group4)

- **Verdict.** Tried to move the fp4 GEMV dequant off the FMA critical path
  via shared memory (the register double-buffer spilled in round 6): a
  same-warp ping-pong (decode g+1 -> shared, FMA g from shared) and a
  producer/consumer warp split (threadIdx.z role, RING=3 SPSC ring, producer
  2 groups ahead, consumer issues zero shuffles). Both bit-exact vs the
  shipped kernel (rel-err 2.74e-3), both 2.5-3.3x slower (14-17% roof vs
  group4's 43%, contended H20): the STS+LDS round-trip and per-group
  `bar.sync` cost more than the shuffle issue they remove, and the LDS
  latency lands on the critical path. group8 ties (same issue/elem, more
  regs). bf16 IO was already done in round 1. The register group4 stays; the
  46%->57% gap to the nodecode floor needs fewer dequant instructions/elem,
  not a different buffer. Entry:
  `docs/experience/errors/2026-08-25-gemv-shared-pingpong-rejected.md`.

## 2026-08-25 — phase exit: final 80/3800 bench — decode 49 / prefill 1172 tok/s, not met; gap is dequant issue throughput + GDN chunk, not physics

- **Verdict.** Final measurement at HEAD ea8ba7f (f32 scales, grouped dequant)
  in a fully-idle H20 window (all 8 GPUs at 0%, BW 3312). Slice4 (3 GDN + 1 FA,
  the 27B's exact 3:1 mix), graph-captured: 1.828 ms/tick decode, 0.0557 ms/tok
  prefill-512. Extrapolated with lm_head (0.5195 ms, 55.4% roof) and fixed cost
  counted once: decode 20.41 ms/tok (49.0 tok/s) vs the 80 target (1.63x gap),
  prefill 0.853 ms/tok (1172 tok/s) vs 3800 (3.2x). Rooflines: decode 162-196
  tok/s (20.4 GB at 3.3-4.0 TB/s), prefill 5898 tok/s (25.7 TFLOP at 296
  TFLOPS) — both targets are 41-64% of roof, physics allows them. The decode
  gap is the fp4 GEMV dequant issue throughput (direct kernel 46% roof, lm_head
  55% — the bf16 GEMV hits 42-116%, so the dequant is the cap); the prefill gap
  is the GDN serial scan (27.4% of the tick, WY rejected 2.6x). Entry:
  `docs/experience/wins/2026-08-25-gemv-instr-gdn-wy.md`.

## 2026-08-25 — accept-or-reject: fp4 packed scales f32 -> e4m3 — REVERTED (5-11% decode regression, not neutral)

- **Verdict.** The internal fp4 per-32-block scale changed from f32 to e4m3fn
  bytes (uint8 view) — the checkpoint's native scale dtype, 4x less scale
  traffic. `pack_fp4` rounds block_max/6 to e4m3; the sm90 fp4 kernels + CPU
  kernel decode in-register (integer bit-trick, no exp2). CUDA parity 31/31
  green. Per-linear decode GEMV is 7-11% slower (issue-bound: the decode
  instructions cost more than the 29% traffic savings), and the regression
  surfaces end-to-end: slice4 decode 1.828 -> 1.937 ms/tick (+6.1%, lm_head
  alone is 28% of the wall at +11%). An earlier "neutral" measurement
  (1.841 ms) was an anomaly, inconsistent with the per-linear data. Prefill
  neutral (compute-bound). Reverted in ea8ba7f; the e4m3 work stays in git
  history. Entry: `docs/experience/wins/2026-08-25-fp4-e4m3-block-scales.md`.

## 2026-08-25 — accept-or-reject: chunkwise-WY GDN prefill — REJECTED (2.6x slower than serial scan)

- **Verdict.** Ported the tilelang branch's chunkwise-WY prefill pair
  (`qwen36_prefill_wy.py` + `qwen36_prefill_scan_o.py`, unmerged
  `feat/qwen36-gdn-megakernel`) to replace the serial GDN prefill chunk kernel
  (27.6% of the prefill-512 tick). Correctness green (parity vs
  `reference.gdn_forward`, rtol=1e-2; T=1 cross-check vs decode kernel) but
  **2.6x slower** on the H20 pod: serial 4.73 ms vs WY 12.38 ms (A=1.62 +
  B=10.76). Root cause: WY is O(T*C*K) (C=64, 64x more FMAs) vs the serial
  O(T*K), and the serial kernel is compute-bound (64KB state fits in L2, 48
  heads saturate the SMs) — the WY's chunk-parallelism has nothing to repay
  its extra compute and two-launch/24MB-intermediate overhead. Reverted to the
  serial kernel. The WY solve math is sound (decay-first delta rule, verified
  1e-17) and the port is in git history if a memory-bound or few-head config
  appears. Entry: `docs/experience/errors/2026-08-25-gdn-prefill-wy-rejected.md`.

## 2026-08-25 — accept-or-reject: fp4 GEMV grouped dequant — direct-call 42% -> 46% roof, slice4 decode 1.887 -> 1.837 ms/tick

- **Verdict.** The dequant is grouped 4 micro-tiles at a time: load 4, decode
  all 4 (32 shuffles), then FMA all 4 — hoisting every shuffle off the FMA
  critical path so its latency hides behind the FMA dependency chain.
  Direct-call roofline +9.5-10.9% on both shape orientations (42.0 -> 46.0%,
  38.5 -> 42.7%); slice4 decode 1.887 -> 1.837 ms/tick (530 -> 545 tok/s).
  Rejected: register double-buffer (spills, 22%), 6-op bitcast (32%), 256-entry
  byte-LUT (26%, gather loads), no-X-buffer (19%), 2 accumulators (no help).
  Entry: `docs/experience/wins/2026-08-25-fp4-gemv-grouped-dequant.md`.

## 2026-08-25 — accept-or-reject: fp4 GEMV vectorized dequant — slice4 decode 6.94 -> 1.89 ms/tick (3.67x), big projections 24-33% -> 30-54% roof

- **Verdict.** The fp4 GEMV dequant (the gap named in the 80/3800 verdict
  above) is vectorized: warp-shuffle LUT decode (1 op/elem vs 9 for the
  bitcast) + partial-scale (`acc += s * sum(X*w)`, 1 FP op/elem vs the 2-mul
  chain). The nodecode floor jumped 30% -> 57% roof; the shipped lutshfl
  reaches 44% (78% of floor). Slice2 decode 1.922 -> 1.285 ms/tick (1.50x),
  slice4 6.941 -> 1.893 (3.67x vs WGMMA). The big projections (17408x5120
  class) are capped at ~30% roof by the backend dispatch overhead (~0.022
  ms/call), not the kernel — the direct kernel is at 44%. lm_head (large N,
  overhead amortized) hits 54%. Entry:
  `docs/experience/wins/2026-08-25-fp4-gemv-vectorized-dequant.md`.

## 2026-08-25 — accept-or-reject: final 80/3800 bench — not met (35.5 decode / 992 prefill tok/s), gap is fp4 dequant efficiency, not physics

- **Verdict.** Final measurement at HEAD c97f79c after the bf16 GEMV and
  native FP8 levers, in a fully-idle H20 window (all 8 GPUs at 0%). Slice4
  (3 GDN + 1 full-attn, the 27B's exact 3:1 mix), graph-captured:
  2.662 ms/tick decode, 0.0654 ms/tok prefill-512. Extrapolated with lm_head
  (0.890 ms, measured) and fixed cost counted once: decode 28.2 ms/tok
  (35.5 tok/s) vs the 80 target (2.25x gap), prefill 1.008 ms/tok
  (992 tok/s) vs 3800 (3.8x). The 2026-08-24 all-levers entry's 16.9 tok/s
  decode was an extrapolation artifact (GDN-only per-layer x 64 overcounts
  lm_head 32x); the corrected method on its own data gives 32.9 tok/s — the
  kernel delta since is ~8% (fp8 GEMV decode). Roofline: decode 162-196
  tok/s (20.4 GB at 3.3-4.0 TB/s), prefill 5898 tok/s (25.7 TFLOP at 296
  TFLOPS) — both targets are 41-64% of roof, physics allows them. The gap
  is code: the fp4 GEMV dequant stage caps decode at 24-32% of roof (the
  bf16 GEMV on the same schedule hits 42-116%), and the fp4 MLP prefill
  path runs at 21% of peak (62% of the tick). Eager per-op event spans
  overcount the M=1 tick 3.6x — the graph wall is the metric. Entry:
  `docs/experience/wins/2026-08-25-bf16-gemv-fp8-weights.md`.

## 2026-08-25 — accept-or-reject: native FP8 weights shipped — GDN prefill 1.48x, decode 1.05x, but the 3800 target needs the MLP lever too

- **Verdict.** `load_hf` now keeps the checkpoint's FP8 GDN projections native
  (`<key>.w8` e4m3 + `<key>.wscale` per-128-block, bf16 master recording-only;
  `cfg.fp4` packing skips them) instead of dequantizing to bf16 and re-packing
  to fp4. sm90 dispatch: `make_linear_fp8_mma` (deepgemm 2xAcc, per-128-block
  accumulator scaling, no K-loop dequant) for M>1 prefill, `make_linear_fp8_gemv`
  for M=1 decode. Parity green (local 72, pod CUDA 31). Isolated GDN bench
  (same-process back-to-back, contention-independent): fp4 5.657 ms (62.6
  TFLOPS) → fp8 3.819 ms (92.8 TFLOPS), **1.48x prefill**; decode graph
  2.799 → 2.672 ms/tick, **1.05x**. The 3800 tok/s target is not met by this
  lever alone: the GDN is ~23% of prefill FLOPs, the MLP (NVFP4, unchanged fp4
  path) is ~77% — its dequant efficiency is the remaining lever. Entry:
  `docs/experience/wins/2026-08-25-native-fp8-weights.md`.

## 2026-08-25 — accept-or-reject: bf16 GEMV shipped, but it is NOT the 27B decode lever (premise correction)

- **Verdict.** `make_linear_bf16_gemv` shipped (sm90, parity green local +
  pod CUDA): the fp4 GEMV schedule minus the dequant, 42-116% of HBM roof on
  the bf16 projection shapes vs 20-42% for the padded-M=16 WGMMA it replaces
  (1.9-3.9x). But the SOTA-all-levers bench's claim that "GDN projections are
  bf16, 73% of decode bytes" was wrong: on `cfg.fp4=True` `load_hf` packs
  EVERY projection (GDN `in_proj_*`/`out_proj`, MLP, `lm_head`) to fp4, and
  the decode per-op profile shows only `linear_fp4` (86% of GPU), zero bf16
  `linear` ops. Slice decode before/after: 1.932 → 1.922 ms/tick (noise). The
  bf16 GEMV serves non-fp4 models, not the 27B. The real 27B decode lever is
  fp4 GEMV efficiency (24-33% roof — the dequant stage, since the bf16 GEMV
  on the same schedule hits 42-116%). Entry:
  `docs/experience/wins/2026-08-25-bf16-gemv-decode.md`.

## 2026-08-24 — phase exit: SOTA kernel round complete — 80/3800 not met, gap is kernel efficiency not physics

- **Exit verdict.** Final bench on real NVFP4 slices after all levers (fp4
  GEMV decode, fused GDN decode + chunk prefill, multi-block norm/act, graph
  capture, fp8 prefill WGMMA, FlashAttention paged attn). Slice4 (3 GDN + 1
  full-attn, the 27B's exact 3:1 layer mix) extrapolates to the full model
  with no mix correction: decode 59.3 ms/tok (16.9 tok/s) vs 80 target,
  prefill 1.02 ms/tok (976 tok/s) vs 3800 — 4.7x / 3.9x off under a 99%-util
  co-tenant (high-contention phase: ~15 / ~487 tok/s). Roofline says the
  targets are physics-allowed: decode roof 129 tok/s (30.9 GB at 4 TB/s,
  target = 62% of BW roof), prefill roof 3835 tok/s mixed-dtype / 5317
  fp8-everything — the 3800 target IS the mixed-dtype roofline at 100%
  tensor utilization, unreachable while `load_hf` dequantizes the
  checkpoint's FP8 GDN projections to bf16. The gap is code: bf16 M=1
  projections at ~10-15% of BW roof (73% of decode bytes) need a bf16 GEMV
  path; prefill needs fp8 weight retention plus in-engine fp8 GEMM efficiency
  (16-22% of peak contended, 60-80% isolated). Contention finding: eager
  dispatch is pathological on a shared GPU (784 us/op queue wait, 27.2 ms
  wall vs 2.1 ms GPU sum) — graph capture is the only viable decode mode.
  Entry: `docs/experience/wins/2026-08-24-sota-all-levers.md`.

## 2026-08-24 — default flip: paged_attention on sm90 is FlashAttention (was serial-scalar)

- **Default flip.** `paged_attention` in the sm90 cell is now
  `kernels_mma.make_paged_attention_mma` — the FlashAttention online-softmax
  schedule ported to paged KV + GQA, bf16 IO, block_M 16 (decode) / 64
  (prefill). The f32 serial-scalar kernel stays in kernels.py as the
  cpu/metal floor. Kernel-level at the 27B full-attn shapes (H=24, Hkv=4,
  D=256): decode M=1 KV=4096 37.84 → 0.456 ms (83x), prefill M=512 1056 →
  0.062 ms (17100x). Prefill is 35% of the bf16-tensor roofline under a
  99%-util co-tenant (within 2x idle); decode is still ~30x off the memory
  roofline — tilelang 0.1.13 lowers the paged gather to synchronous loads
  (ponytail: split-KV flash-decoding with pipelined gathers). Full-model
  impact: 16 full-attn layers add ~7.3 ms/tick decode (contended), ~0.002
  ms/tok prefill. Entry: `docs/experience/wins/2026-08-24-paged-attention-fa.md`.

## 2026-08-24 — default flip: fp8 prefill path on sm90 (e4m3 activations + fp4->e4m3 WGMMA)

- **Default flip.** `linear_fp4` with M>1 on sm90 now runs fp8 WGMMA (e4m3
  activations, e2m1fn→e4m3 weight dequant in the K-loop, f32 accumulate)
  instead of bf16 WGMMA — 1.5x on the slice prefill tick (6021 → 7839 tok/s,
  extrapolated 268 → 399 tok/s). pack_fp4's block scale moves per-16 → per-32
  to match the fp8 WGMMA K-tile (one scale per tile, no temp-fragment
  epilogue). e4m3's ~2% multiplicative quant error does not average down over
  K, so the parity gate uses an identical-quant torch reference. 399 tok/s is
  9.5x off the 3800 target — the kernel is dequant-bound (e4m3 cast in the
  K-loop, ~20% of fp8 peak); production fp8 GEMMs precompute fp8 weights.
  Entry: `docs/experience/wins/2026-08-24-fp8-prefill-wgmma.md`.

## 2026-08-24 — phase exit: decode tick captured (CUDA graph) on sm90

- **Exit + default flip.** The decode tick is now a captured kernel sequence
  (design-engine.md invariant): `_DecodeGraph` captures `model.forward` once
  per batch-size bucket (day-1 M=1) and replays per token — auto-on for CUDA,
  eager the default elsewhere and the fallback on capture failure. Dispatch
  drops from 899 ops x 20.4 us = 18.3 ms (full-model extrapolation) to
  0.040 ms (36 us pinned async copies + 3 us replay) on the 2-GDN-layer
  slice; the replay cost is op-count-independent. Two prerequisites landed
  with it: a `write_tokens` sm90 scatter kernel (the pool's host loop had
  per-token GPU->CPU syncs, uncapturable) and an on-device `_inv_freq` cache
  (the CPU-cached tensor H2D-copied on every rope, illegal in capture).
  Parity: eager vs captured token streams identical (tiny model, CUDA),
  `test_ops_parity.py` 26/26 on CUDA. Entry:
  `docs/experience/wins/2026-08-24-decode-graph-capture.md`.

## 2026-08-24 — verdict: bf16 IO + bitcast fast decode accepted on sm90

- **Verdict.** The sm90 MMA kernels (3 gemms + `linear_fp4_mma` +
  `linear_fp4_gemv`) switch from f32 to bf16 IO (bf16 WGMMA, f32 accumulate),
  and the e2m1fn decode switches from 4x `exp2` to integer bit-pattern
  synthesis (`sign<<31 | (126+e)<<23 | m<<22`, reinterpreted as float —
  ties the warp-shuffle LUT, no warp-cooperation constraint). Big-shape
  GEMV 1.8-2.1x faster (17408x5120: 0.138 -> 0.077 ms, 15% -> 27% of roof;
  lm_head 1.85 -> 0.89 ms, 33% of roof); WGMMA path 1.5x; slice decode
  4.55 -> 3.91 ms/tick. CUDA parity 25/25 at rtol=1e-2. Entry:
  `docs/experience/wins/2026-08-24-fp4-gemv-bitcast-bf16.md`.

## 2026-08-24 — verdict: multi-block norm/activation accepted on sm90

- **Verdict.** `silu_mul` gridded over M (1024-element chunks) and
  `rmsnorm` split-K (per-chunk partial sums + apply, two launches) in the
  portable floor — same source on CPU/CUDA/Metal, serial fragment-scalar
  accumulators (the example `T.reduce_sum` idiom is not Metal-portable).
  Slice prefill 512: silu_mul 46.1 -> 0.07 ms, tick 119.6 -> 73.7 ms
  (4257 -> 6886 tok/s). Decode rmsnorm 0.445 -> 0.410 ms — now
  launch-bound at 2 launches/call; next lever is fusion, not more blocks.
  CUDA parity 25/25. Entry:
  `docs/experience/wins/2026-08-24-multiblock-norm-act.md`.

## 2026-08-24 — phase exit: GEMV + chunk-kernel round closed on sm90

- **Exit.** Both kernels default-on, final slice numbers on H20: smoke
  8-token average 48.85 -> 31.09 -> 5.46 ms/tok, decode-only 5.335 ms/tick
  (187 tok/s), prefill-512 0.2226 ms/tok (4491 tok/s, 3800 slice target
  met). Full-model extrapolation (lm_head corrected): ~102 ms/tok decode
  (9.8 tok/s) and 6.95 ms/tok prefill (144 tok/s) vs 80/3800 targets —
  8.2x / 26x gap. Next levers per the profile: launch count (899 ops/tick,
  20 ms dispatch) and the single-block `silu_mul` grid (40% of prefill),
  not new GEMM schedules. Entry:
  `docs/experience/wins/2026-08-24-gemv-chunk-kernels.md`.

## 2026-08-24 — default flip: sm90 GDN prefill uses the fused chunk kernel

- **Flip.** `linear_attn_chunk` on sm90 now dispatches prefill (T>1) to
  `make_gdn_chunk_fused` (one launch per value head: conv1d + SiLU +
  q/k-norm + decay-first delta recurrence + gated RMSNorm + z-gate, serial
  scan over T) instead of the torch-eager reference (~150k tiny launches
  per 512-token prefill). T=1 keeps the decode kernel. 27B slice prefill:
  11.01 -> 0.2212 ms/tok (49.8x) on H20, JIT-free — the 3800 tok/s slice
  target is met. CUDA parity 25/25. Entry:
  `docs/experience/wins/2026-08-24-gdn-prefill-chunk.md`.

## 2026-08-24 — verdict: sm90 fp4 GEMV decode accepted (CUDA-verified)

- **Verdict.** `make_linear_fp4_gemv` (733cbcd, SOTA copy of
  `example_dequant_gemv_fp16xint4.py`) accepted as the sm90 M=1 decode
  path: CUDA parity 23/23, GEMV beats WGMMA-padded on every fp4 linear
  (1.3–5.9x), slice decode 10.577 -> 5.452 ms/tick (94.5 -> 183.4 tok/s)
  on H20. Roofline 12–14% on big shapes — headroom in decode ALU and
  launch count, not weight streaming. Entry:
  `docs/experience/wins/2026-08-24-fp4-gemv-decode.md`.

## 2026-08-24 — default flip: sm90 GDN decode uses the fused megakernel

- **Flip.** `linear_attn_chunk` on sm90 now dispatches decode (T=1) to
  `make_gdn_decode_fused` (one launch per value head: conv1d + SiLU +
  q/k-norm + decay-first delta recurrence + gated RMSNorm + z-gate, ported
  from `examples/gdn/qwen36_gdr_decode_fused.py` @ tilelang branch
  `feat/qwen36-gdn-megakernel`) instead of the torch-eager reference
  (~384 tiny launches/layer/tick). Prefill (T>1) keeps the reference.
  27B slice decode: 65.46 -> 47.16 ms/tok (28%) on H20, JIT-free.
  Entry: `docs/experience/wins/2026-08-24-gdn-decode-fused.md`.

## 2026-08-24 — default flip: sm90 cell switches from naive FMA to MMA (WGMMA)

- **Flip.** The sm90 cell now uses the MMA schedules in `kernels_mma.py`
  (shared-memory tiled `T.gemm` + pipelined K-loop, ported from
  `examples/gemm/example_gemm.py` and the Hopper dequant+gemm example) for
  `gemm_{nt,nn,tn}` and `linear_fp4`. The naive FMA schedules stay in
  `kernels.py` as the metal/other-arch fallback. 27B slice decode:
  1180.19 -> 48.85 ms/tok (24x) on H20, JIT-free. Entry:
  `docs/experience/wins/2026-08-24-sm90-mma-gemm.md`.
- **Format.** `pack_fp4`/`unpack_fp4` switched from the OCP e2m1 LUT (with
  zero) to e2m1fn (no zero) to match the kernel decode and the Hopper SOTA;
  `dequant_nvfp4` keeps the OCP grid (checkpoint wire format is separate).
  CUDA parity 21/21; CPU suite 64 passed, 1 skipped.

## 2026-08-24 — verdict: sm90 CUDA target accepted; real 27B slice forwards + trains

- **Verdict.** sm90 cell accepted: the 2-layer GDN slice of Qwen3.6-27B
  NVFP4 forwards through the engine (8 tokens, 1180.19 ms/tok JIT-free) and
  takes a training step (22.0 s/step, loss 11.2405 → 10.6960) on an H20,
  with CPU/CUDA logits parity to 6 decimals. 60 passed on CUDA. Entry:
  `docs/experience/wins/2026-08-24-sm90-real-slice.md`.
- **Fixes.** ModelOpt NVFP4 global scale is stored reciprocal (divide, not
  multiply); ModelOpt FP8-block `weight_scale_inv` is the scale itself
  (multiply, not divide) — both confirmed against agent-infer
  `quant_format.rs`, hermetic loader tests updated. `qwen36_27b()` is
  untied (checkpoint ships `lm_head.weight`). Bench warmup uses the timed
  `prompt_len` (JIT specializes per shape; a shorter warmup leaked NVCC
  into the measurement). sm90 registered with the naive FMA gemms;
  `Backend.device` pins `cuda:<current>`.
- **Bench (H20, tiny).** prefill 0.507 ms/tok (1973.5 tok/s), decode
  3.624 ms/tok (275.9 tok/s), prompt_len=128, gen=32, JIT-free.

## 2026-08-24 — format loaders: official NVFP4, per-tensor FP8, AWQ-int4; 23MB fixture retired

- **Features.** `load_hf` gains three formats (all dequant to bf16 at load):
  official NVIDIA NVFP4 naming (`weight` u8 nibbles + `weight_scale` f8 +
  scalar `weight_scale_2`; reuses the ModelOpt e2m1 math), per-tensor FP8
  (f8 `weight` + scalar `weight_scale` — the official-NVFP4 GDN/attn path and
  standalone FP8), and AWQ-int4 (`qweight`/`scales`/`qzeros`, autoawq GEMM
  packing, group size from `quantization_config`). `dequant_awq` added to
  `ops/reference.py`. Five formats now covered: bf16 HF, MLX-4bit, ModelOpt
  NVFP4/FP8-block, official NVFP4, FP8, AWQ-int4.
- **Tests.** 64 passed, 1 skipped on CPU (was 62+1). New synthetic
  per-format tests in `tests/test_weights.py` (KB-sized, formula-reference,
  `torch.equal`): `test_nvfp4_official_load`, `test_awq_load`,
  `test_mlx_affine_load`. Deleted: the 23MB `tests/fixtures/
  qwen35-2layer-mlx4/`, `tests/test_real_weights.py`, `scripts/crop_fixture.py`,
  and the orphaned `qwen35_08b` config. Entry:
  `docs/experience/wins/2026-08-24-format-loaders.md`.

## 2026-08-24 — pretrain loop + save_hf; ruff format gate turned on

- **Features.** `save_hf(model, path)` (model.py): HF safetensors + config.json
  roundtrip with `load_hf` — tensor names are the reverse of `_LAYER_SUFFIXES`,
  fp4 masters saved bf16 and re-packed on load. `pretrain(...)` +
  `JsonlDataset` (train.py): JSONL `"text"` corpus → eos-separated packed
  sequences, causal-LM loss via `train_step`, `cosine_warmup` wired in (its
  first production caller), seeded epoch-wise shuffle, periodic + final
  checkpoints. CLI: `tilerl pretrain --model tiny --data <jsonl> --steps N
  --seq-len 512 [--ckpt-dir D] [--ckpt-every M] [--lr] [--warmup] [--seed]`.
- **Default flip.** `ruff format --check` is a blocking CI gate
  (`continue-on-error` removed); tree reformatted (19 files), `ruff check
  --fix` clean.
- **Tests.** 60 passed, 3 skipped on CPU and Metal (was 58+3). New
  `tests/test_pretrain.py`: JSONL packing/padding, pretrain finite-loss +
  params-moved + checkpoint landing, save_hf → load_hf forward-match.

## 2026-08-24 — consolidation: -20% src LOC, decode exact, real weights coherent

- **LOC.** src 5671 → 4536 (-20.0%, hard target ≤4537 met); tests 1503 lines.
  Deletions: 7 selfcheck mains (-758), `weights.py` merged into `model.py`,
  `ops/fp4.py` merged into `ops/reference.py`, `testing.py` __getattr__
  delegation (-52), duplicated param-key mappers merged, docstring tightening.
- **Correctness fixes.**
  - **MLX 4-bit dequant formula** was `s*(q-b)`; the MLX quantized-matmul kernel
    uses `s*q+b` (scales may be negative). Real 0.8B next-token was gibberish,
    now exact (" Paris" for "The capital of France is").
  - **GDN conv1d carry** was zero-padded at every decode step (documented day-1
    limitation). `conv_window` [B,K-1,qkv_dim] now threads through
    `gdn_forward`/`LinearStatePool`/engine prefix snapshots; segmented decode is
    bit-exact vs one-shot prefill (parity test
    `test_gdn_conv_window_makes_step_exact`). Full 24-layer generation is
    coherent.
  - **Tape `add` handler was missing** — residual adds were unrecorded, so only
    embed/final_norm received grads (layers trained nothing). Caught by the new
    production-model gradcheck (AGENTS.md gate); fixed with a 2-line handler.
- **Features.** `load_hf(cfg, source, num_layers=N)` truncation (full-attn
  subset matched, skipped layers not required); precision×arch dispatch
  registry in `ops/backend.py` (`(precision, arch) -> kernel set`, fallback
  exact→any→any, GPU slots registered empty → NotImplementedError);
  `docs/support-matrix.md`.
- **Tests.** 55 passed, 3 skipped (GPU + 2 real-weight env-gated). New:
  conv-carry exactness, cosine_warmup, clip_grad_norm, production-model
  gradcheck, opd_loop smoke, white-box prefix-snapshot, engine miss-path,
  MLX dequant (via real-weight test), load_hf truncation.
- **Real weights (env-gated `TILERL_TEST_REAL=1`).** Qwen3.5-0.8B-MLX-4bit:
  truncated 2-layer forward + train_step pass; full 24-layer generation
  coherent ("The capital of France is Paris, and the capital of the United
  States is Washington, D.C.").
- **Bench (CPU, tiny).** prefill 45.68 ms/tok, decode 3.17 ms/tok
  (prompt_len=128, gen=32). Decode is far under the 60 ms/tok baseline — the
  conv-carry fix removed the per-step re-prefill penalty; prefill includes
  one-time JIT.
- **Pending-remote.** GPU arches (sm90/sm100/rocm/metal) registered as empty
  sets — NotImplementedError on use. 27B download test stays pending-remote.

## 2026-08-24 — phase exit: integration green, tiny model baseline

- **Full test suite green.** 45 passed, 1 skipped (test_gpu_targets — no CUDA
  on this host). Covers ops parity (tilelang vs torch-eager reference), KV
  pool lifecycle, e2e generation/prefix-cache/training/gradcheck/fp4, and the
  OpenAI-compatible server (health, models, non-stream, SSE stream).
- **Integration fixes.** Tape records structural ops (reshape/transpose/
  slice/add) as first-class entries so the id()-based grad chain never breaks;
  GDN layer is one monolithic backend op with a gradchecked torch-eager
  backward; dense training path (`kv.dense=True`) bypasses paged attention;
  fp4 pack/unpack moved to `ops/fp4.py`; engine prefix cache publishes all
  block-aligned prefixes; engine `_loop` guards against silent thread death.
- **Tiny model baseline (CPU).** prefill 56.98 ms/tok (17.6 tok/s), decode
  60.03 ms/tok (16.7 tok/s), prompt_len=128, gen=32. See
  `docs/experience/wins/2026-08-23-tiny-model-baseline.md`.
- **Smoke.** `tilerl --help`, `bench`, `train --steps 20`, `serve --model tiny`
  all run end-to-end. Server SSE streaming verified via curl.
- **Pending-remote.** GPU targets (cuda/rocm/metal) compile from the same
  kernel source but are unverified (no GPU on this host). 27B weights not
  downloaded (`TILERL_QWEN38_SOURCE` placeholder).

## 2026-08-23 — bootstrap: project scaffold and contract

- **TileLang-only backend.** One kernel source targeting cpu/cuda/rocm/metal;
  the numpy backend framing from the earlier partial bootstrap was deleted.
- **CPU target is the portable default and CI path.** This machine has no GPU;
  all verification runs on CPU. GPU targets are pending-host.
- **torch reduced to tensor container.** No `torch.autograd` / `torch.optim`;
  training runs on a hand-written reverse-mode autograd tape mirroring
  `agent-infer/crates/autograd`.
- **Engine seam**: submit/poll + `StepLimits`, continuous batching, one
  forward per tick. State: paged KV (full attention) + recurrent state
  (gated-delta) + hash prefix cache.
- **OPD training** shares the engine and weights with serving.
- **OpenAI-compatible server** entry point (`tilerl serve`).
- Docs scaffold: `AGENTS.md` (canonical agent contract; `CLAUDE.md` symlinks
  to it), `README.md`, bench-entry template under `docs/experience/wins/`.
