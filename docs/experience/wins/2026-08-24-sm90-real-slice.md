# sm90 CUDA target: real 27B slice forwards and trains — H20, 2026-08-24

> Status: Shipped

## Context

First real-weight run on GPU. The 2-layer GDN slice of Qwen3.6-27B NVFP4
(`/host/tc27-nvfp4-slice2`, both layers GDN) must forward through the engine
and take a training step on sm90, with CPU/CUDA logits parity as the
correctness gate. The metric that matters: does the real checkpoint produce
non-uniform logits and a dropping loss.

## What Worked

- **sm90 cell registered** with the naive FMA gemms (the Metal schedule —
  CUDA's MMA lowering rejects global operands and requires tile M/N divisible
  by 16). `Backend.device` pins `cuda:<current>`. Data-race check disabled
  for CUDA (per-thread fragment false positive, same as CPU/Metal).
- **Two loader scale-semantics bugs found via the real slice.** The slice
  produced uniform logits (loss = ln(vocab) = 12.4225, all-zero tokens).
  Diagnostic: weight absmean 266755 — weights ~1e7 too large. Root causes,
  both confirmed against `agent-infer/crates/quantized` (ScaleApply::Divide
  line 225, Multiply line 165; `qwen35_loader.rs:1424-1429`):
  - ModelOpt NVFP4 `weight_global_scale` is stored as the **reciprocal** —
    dequant must divide, not multiply (1/6278, not ×6278).
  - ModelOpt FP8-block `weight_scale_inv` is the **scale itself** despite the
    name — dequant must multiply, not divide (×0.000243).
  Both fixed in `ops/reference.py` with hermetic synthetic tests updated.
- **`qwen36_27b()` was tied** (`tie_word_embeddings=True`, inherited from
  qwen38); the checkpoint is untied and ships `lm_head.weight`. Flipped.
- **CPU/CUDA parity holds**: logits std 0.636383 on both targets, first-8
  values identical, per-layer rmsnorm norms identical (7.74 / 109.7 / 83.5 /
  130.9 / 78.2421).
- **Training step drops loss**: 11.2405 → 10.6960 after one AdamW step.
- **Bench warmup fix**: warmup now uses the timed `prompt_len` — JIT
  specializes per shape, so a len-4 warmup leaked ~10s of NVCC into the
  timed prefill (fake 78.768 ms/tok). Clean numbers below.

## Rule

A new quant format is not loaded until its scale semantics are checked
against the reference loader (agent-infer `quant_format.rs`): the tensor
name (`scale` vs `scale_inv` vs `global_scale`) does not tell you whether it
multiplies or divides — the producer's code does. And JIT-specialized
backends must warm up at the exact timed shape.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | 4e85cb5 | H20 | cuda/sm90 | tiny | 0.507 | 3.624 | 1973.5 prefill / 275.9 decode |
| 2026-08-24 | 4e85cb5 | H20 | cuda/sm90 | 27B slice (2 GDN layers) | — | 1180.19 | 0.85 decode |

Real slice, JIT-free steady state (after same-shape warmup): generate 8
tokens in 9441.5 ms (1180.19 ms/tok); `train_step` 22.0 s/step (loss
11.2405 → 10.6960). Load 24.3 s, 65 parameter tensors. First train_step
includes backward-kernel JIT (~30-120 s per new shape/dtype — the 22 s
figure is step 2, JIT-free).

Day-1 ceilings (known, marked in code): naive FMA gemms (~100-200 GFLOPS vs
H20's ~60 TFLOPS FP32 — decode is dominated by the lm_head gemm's
block_M=64 over-compute on M=1); fp4 scalar dequant inside `linear_fp4`;
GDN backward is torch-eager; tilelang eager JIT has no disk cache, so every
new (shape, dtype) pays NVCC once per process.

Raw artifacts: `/work/smoke3.log`, `/work/bench_cuda2.log` (pod, H20).
