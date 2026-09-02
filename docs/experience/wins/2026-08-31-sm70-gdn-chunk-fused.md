# sm70 GDN chunk fused: prefill 64.9s → 15.1s — V100/sm70, 2026-08-31

> Status: Shipped

## Context

The B=8 decode breakthrough (31.8 t/s) left the prefill as the dominant cost:
tick 1 was ~65s for 8 short prompts (~5 tokens each). Profiling
(`scripts/prof_decode_tick.py`) showed `linear_attn_chunk` at 62.65s of the
64.24s tick (97.5%) — the gated-delta chunked forward. Root cause:
`gdn_chunk_fused` was only registered in `_SM90_KERNELS`, so sm70 fell back
to `reference.gdn_forward` — a Python serial scan over T=512 with ~10 eager
PyTorch ops per step, ~250k ops per prefill.

## What Worked

- **Register `gdn_chunk_fused` for sm70** (one line in `registry.py`): the
  kernel is target-neutral TileLang (block-parallel, no warp specifics), same
  source as sm90. It compiles and runs on V100 without modification.
  Prefill: 64.9s → 15.1s warm-cache (4.3×).
- **M=32 GEMV for the prefill M>8 path** (`linear_fp4_gemv_sm70_m32`):
  the C extern `tl_fp4_gemv_tiles_f16_m<G,M>` is already templated on M, so
  extending from M=8 to M=32 is a registry entry + a dispatch change.
  M=512 prefill: 16 launches/layer instead of 512 (32× fewer). Parity PASS
  on 12 real projections (M=1/8/16/32, worst 6.17e-4, gate 1e-2).
- **`build_engine(kv_tier_path=...)`**: wire the SSD KvTier (was `tier=None`
  by default — the class existed but was unreachable from the factory).

## The profiling trap

The first profiling run showed `linear_fp4` at 0.97s (1.5%) and
`linear_attn_chunk` at 62.65s (97.5%). The M=32 chunking was optimizing the
wrong 1.5%. The actual bottleneck was a missing kernel registration — a
one-line fix, not a kernel optimization. **Profile before optimizing.**

A second trap: `sample_batch` appeared to take 3.88s (243ms/call). Its
`.tolist()` is a GPU→CPU sync that absorbs ALL prior forward GPU time, so
the timing reflects the forward, not the sampling. Greedy argmax itself is
~1ms. Backend method timings in eager mode are wall-clock, not GPU-time —
the `.tolist()` sync is where the GPU cost lands.

## _PREFILL_BUCKET=16 was slower

Reducing the bucket from 64 to 16 (M=512→128) made the prefill 18.5s (worse
than 15.1s). The GDN serial scan is faster at T=512 than T=128 on V100 —
the fixed per-call overhead dominates at smaller T, and the kernel's
occupancy is already low (one block per batch×head, ~6% SM utilization).
Larger T amortizes the launch overhead better. Reverted to 64.

## Rule

A kernel that exists in the registry but isn't registered for your arch
silently falls back to the torch-eager reference — which is a Python serial
scan, not a GPU kernel. The symptom is a prefill that takes 100× longer
than the decode for the same model. Check `_resolve(precision, arch)` for
every fused kernel the hot path calls, not just the ones you wrote.

## Results

| date | commit | machine | target | model | B | prefill (warm) | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|---:|
| 2026-08-31 | (prev) | V100 | cuda/sm70 | 27B NVFP4 | 8 | ~65s | 263.8 | 31.8 |
| 2026-08-31 | c199a7e | V100 | cuda/sm70 | 27B NVFP4 | 8 | 15.1s | 260.4 | 31.8 |

Decode unchanged (260ms/tick, 99.1% GPU). Prefill 4.3× faster.

Warm-cache prefill breakdown (tick 1, 8×64 tokens):
- `linear_fp4`: 6.2s (1220 calls, 5.1ms/call — eager Python overhead)
- `sample_batch`: 3.9s (`.tolist()` sync absorbing forward GPU time)
- `linear_attn_chunk`: 3.1s (48 calls, 65ms/call — fused GDN kernel)
- `rmsnorm`: 1.1s (692 calls, 1.5ms/call — eager)
- other: ~0.8s

Raw artifacts: `scripts/prof_decode_tick.py`, `scripts/parity_real_weights.py`
(V100, GPU 0, JIT-cached).
