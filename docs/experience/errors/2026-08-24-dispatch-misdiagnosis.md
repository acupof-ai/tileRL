# 2026-08-24 — Dispatch-overhead misdiagnosis (profile before diagnosing)

## Context

27B-slice decode on H20 was 48.85 ms/tok (2-layer NVFP4 slice), ~460× off the
memory roofline. Asked "why so slow", I read the kernel code and diagnosed
"99% CPU dispatch overhead — tilelang eager per-op Python dispatch, ~1-3ms ×
20-30 calls/tick; CUDA graph capture is the fix". I estimated total GPU work
at ~100-200µs per tick from the WGMMA FLOP counts.

The profile (scripts/profile_slice.py, per-op CUDA events) showed:

| cost | ms/tick | share |
|---|---|---|
| GDN (torch-eager reference path) | 13.1 | 47% |
| embedding re-cast bf16→f32 every tick | 5.85 | 21% |
| linear_fp4 (M=1 padded to 16, WGMMA) | 8.3 | 30% |
| dispatch overhead | 0.66 | 2.3% |

GPU sum was 27.6ms, not 200µs. My diagnosis was wrong on the dominant cost
and off by ~100× on GPU time.

## Root Cause

I assumed GDN ran the tilelang kernel (`make_linear_attn_chunk`). The slice
actually runs `reference.gdn_forward` — the torch-eager fallback — which loops
48 value heads in Python (~384 tiny kernel launches per layer per tick).
Code reading cannot tell you which backend path is live; only the registry
resolution and a profile can. The FLOP estimate was right for the kernels I
read; I never verified those were the kernels running.

## Fix

- GDN decode fused kernel ported from tilelang's `feat/qwen36-gdn-megakernel`
  branch (63×/layer: 6.39 → 0.10ms) — `kernels_mma.py:make_gdn_decode_fused`
- embedding f32 table cached in `backend.py` (5.85ms → ~0)
- Follow-ups: linear_fp4 decode GEMV (33× roofline), prefill chunk kernel
  (97.9% of prefill still torch-eager)

## Rule

离 roofline 差 10× 以上，先 profile 定位再下诊断——读代码不能假设哪条后端
路径在跑，查注册表实际解析到哪个实现。FLOP 估算只对"读到的那个 kernel"
有效；在确认它在跑之前，数字没有意义。
