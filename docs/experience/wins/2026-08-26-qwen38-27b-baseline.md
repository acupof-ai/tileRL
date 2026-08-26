# Qwen3.8-27B NVFP4 full-model serving baseline — sm90, 2026-08-26

> Status: Baseline

## Context

The full-27B reference point that all future decode work is measured against:
the serving-identical build at current HEAD, on the real checkpoint, JIT-warm
and steady-state. Prior numbers were slice extrapolations (2-4 layers); this
is the whole model.

Checkpoint: `/data00/Qwen3.8-27B-NVFP4` (22 GiB — `model.safetensors` 22.6 GB
+ `model_mtp.safetensors` 849 MB, the latter not loaded). Qwen3_5
architecture, 64 layers = 16 full-attn (idx 3,7,…,63) + 48 gated-delta,
untied lm_head.

Serving build (what `tilerl serve --model qwen38-27b` runs): `load_hf` with
`fuse_projections=True`, decode graph ON (auto on CUDA), shipped sm90
kernels. Pool sizes bumped from serving's 256 blocks / 8192 max_total_tokens
to 1024 / 16384 so an 8192-token prompt fits (256 blocks × 16 tokens = 4096);
the per-tick kernels are identical — block-table width is the only difference.
Prefill is chunked at `max_num_batched_tokens=512` like serving, so a 2048 /
8192 prompt is 4 / 16 ticks of M=512.

## What Worked

Two load-path defects blocked this checkpoint at HEAD; both fixed (minimal,
load-path only):

1. **`qwen38_27b()` had 32 GDN value heads; the checkpoint has 48.**
   `A_log` is `[48]`, `in_proj_qkv` is `[10240, 5120]` = (16+16+48)·128.
   Both Qwen3.8 checkpoints on the pod and the Qwen3.6 slices all have 48 —
   the 32 was wrong. `config.py`: `linear_num_value_heads` 32→48.
2. **load_hf's FP8 branch assumed a scalar `weight_scale` (`.reshape(1)`);
   this checkpoint's FP8 linears ship per-channel `[N,1]` BF16 scales.**
   The native-fp8 block-128 `wscale` layout cannot express per-channel scales,
   so `model.py` now dequants per-channel FP8 to the bf16 master (repacked to
   fp4 with the rest when `cfg.fp4`). The scalar path is unchanged.

Result: every linear in the loaded model is fp4-packed (NVFP4 MLP and
per-channel FP8 linears both land on the fp4 path). The native-fp8 path — and
the qkvz fusion (2026-08-26) — is not exercised by this checkpoint; it serves
the separate block-128-scale `/data00/Qwen3.8-27B-FP8` checkpoint instead.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-26 | 628c82d | H20 pod (GPU 7) | cuda/sm90 | Qwen3.8-27B NVFP4, prefill 512 | 0.5136 | — | 1947 |
| 2026-08-26 | 628c82d | H20 pod (GPU 7) | cuda/sm90 | Qwen3.8-27B NVFP4, prefill 2048 | 0.5414 | — | 1847 |
| 2026-08-26 | 628c82d | H20 pod (GPU 7) | cuda/sm90 | Qwen3.8-27B NVFP4, prefill 8192 | 0.5641 | — | 1773 |
| 2026-08-26 | 628c82d | H20 pod (GPU 7) | cuda/sm90 | Qwen3.8-27B NVFP4, decode B=1 | — | 19.03 | 52.6 |

Decode B=1: **19.03 ms/tick, 52.6 tok/s** (avg of 32 steady-state ticks,
graph replay). Prefill: single request, chunked at 512 tokens/tick; ms/tok is
prefill-only (the 1-token decode finish subtracted). Prefill tok/s falls with
prompt length (1947 → 1773) because full-attention chunks attend over the
growing context — the 8192 number is the honest long-context throughput.

Config: `TILERL_TARGET=cuda`, fuse_projections=True, decode graph ON, GPU
`NVIDIA H20` (96 GiB). Load 363 s (CPU dequant of 22 GiB). Peak memory
72.9 GiB (67.9 GiB after engine build). Warmup pass 1 (JIT) 24.4 s, pass 2
(JIT-free) 0.3 s.

## Surprises

- **The serving path could not load this checkpoint at HEAD** — the two
  defects above. Neither was a fallback; both were hard load failures (shape
  mismatch / `reshape(1)` crash). Fixed in this entry.
- **All-fp4 outcome.** The checkpoint's FP8 linears (attn, GDN, lm_head,
  MLP layers 56-63) are per-channel, not block-128 — so they dequant to bf16
  and repack to fp4 rather than staying native-fp8. The model that serves is
  100% fp4 weights; the native-fp8 prefill GEMM and the qkvz fusion are dead
  code for this checkpoint. (The 2026-08-26 qkvz-fusion entry's "Qwen3.8
  checkpoint ships native-fp8 per-128-block" claim describes the separate
  `/data00/Qwen3.8-27B-FP8` checkpoint, not this NVFP4 one.)
- **Memory.** 72.9 GiB peak on a 96 GiB H20. The bf16 masters are
  recording-only (training) — serving carries them as dead weight (~half the
  footprint). A serving-only load that drops the masters would roughly halve
  weight memory.
- **NVFP4 MLP dequant uses `global_divide=True`** (ModelOpt convention — same
  tensor naming as the validated Qwen3.6 slices). Assumed, not independently
  checked against a reference framework's logits.
- **No kernel fell back** to a slower path; decode graph captured and replayed
  (no eager fallback). JIT was fast (24 s warmup) — the persistent
  `/work/tilelang_cache` held most shapes from prior slice runs; only the
  block-table width 1024 was fresh.

## Rule

The full 27B serves at **52.6 tok/s decode B=1** (19.03 ms/tick) and
**1773 tok/s prefill** (8192, chunked 512) on one H20, all-fp4, after a
2-line loader fix — this is the reference point for decode work (target
80 tok/s / 3800 tok/s). Per-channel FP8 scales dequant to bf16 and repack to
fp4; the block-128 native-fp8 path is for the FP8 checkpoint, not this one.

Raw artifacts: `scripts/bench_qwen38_baseline.py` (pod, GPU 7, detached log
`/work/tilerl_baseline.log`).
