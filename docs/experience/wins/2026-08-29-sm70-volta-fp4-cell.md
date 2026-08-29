# sm70 (Volta) fp4 W4A16 decode cell — 2026-08-29

> Status: Shipped. sm70 kernel cell + M>1 dtype fix, parity-gated on a real V100.

## Context

V100 (Tesla V100-SXM2-32GB) is `sm70`. `_arch_for` already returned `"sm70"`,
but `_resolve("fp4", "sm70")` raised `KeyError` — no cell existed, so the engine
could not start on a V100 at all. Volta has none of what the sm90 kernel family
uses: no fp4/fp8 tensor cores, no cp.async, no WGMMA, no TMA. tilelang lowers
`T.gemm` to `mma.sync.m8n8k4` on Volta but **only for fp16** — every one of the
1,406 lines of sm90 MMA schedule is dead here.

Two toolchain facts, verified on the box:
- System nvcc is 11.8 (`-std=c++20` fails: "not defined for option 'std'").
  tilelang hardcodes c++20, so builds must use `/usr/local/cuda-12.4/bin/nvcc`
  (g++ 12.2 backs it). A plain fp16 `T.gemm` then compiles + runs, `max_err 0`.
- Host RAM is 31 GB, ~27 in buff/cache — the 54 GB bf16 master cannot stage,
  and `load_hf(fp4=True)`'s one-shot pack would OOM.

## What Worked

**A minimal Volta cell, one new kernel.** `_SM70_KERNELS = {**_CPU_KERNELS,
"linear_fp4_gemv": make_linear_fp4_gemv_sm70}` + two `_register` calls. The
decode GEMV (`make_linear_fp4_gemv_sm70`) mirrors `make_linear_bf16_gemv`'s
split-K skeleton but reads packed fp4: each thread owns a 16-elem K-slice
(8 natural-layout bytes, one 128-bit load) inside one scale block, decodes
each nibble with the branch-free `_e2m1_fp32` bit-synthesis (2× the LUT path),
and f32-accumulates the bf16 activations. Registered under the standard
`linear_fp4_gemv` name so `_CUDA_PLAN` and `_served_fp4` resolve with no arch
branch; `_served_fp4` skips the sm90 twiddle for sm70 (it reads natural bytes).
M>1 has no sm70 kernel, so `_plan` returns None and the generic f32
`make_linear_fp4` runs.

**Parity on a real V100** (CUDA 12.4 nvcc), vs the f32 fp4 reference:

| M | path | rel err |
|---:|---|---:|
| 1 | `linear_fp4_gemv_sm70` | 1.5e-3 |
| 4 | generic f32 fallback | 1.7e-3 |
| 16 | generic f32 fallback | 1.7e-3 |
| 17 | generic f32 fallback | 1.7e-3 |

All under the `rtol=1e-2` gate. The GEMV also passed standalone at 512², 5120²
(the 27B width), 256×2048.

## The M>1 dtype bug the fallback hid

The M>1 path lands on the generic `make_linear_fp4`, which is **f32-IO**. But
`_rows` casts X to bf16 for every `target.startswith("cuda")` — so the kernel
raised `input X dtype mismatch, expected float32`. sm90 never hit this (it has
an MMA kernel for every M and never reaches the generic path); **sm70 is the
first cuda cell without an fp4 tensor-core kernel**, so it was the first to
exercise the fallback. Fix: `self._f32(x2)` at the generic call site. Only a
single-M parity test would have missed it — it was the M=4/16 sweep that
surfaced it.

## Rule

A new arch cell that reuses a generic kernel must test every M-dispatch bucket,
not just the one its new kernel serves. "CUDA" is not a synonym for "has bf16":
the seven `target.startswith("cuda")` I/O branches assume Ampere+, and the
first pre-Ampere cell to reach any of them finds the latent dtype mismatch.

## Pending-remote

Decode tok/s is not measured yet (the 27B fp4 checkpoint was still quantizing).
The physics: dense 27B ≈ 14 GB/token, V100 HBM2 ~900 GB/s ⇒ ~64 tok/s ceiling.
The GEMV is bandwidth-shaped (stream WQ at 0.5 B/elem, decode, f32 FMA), so it
targets that ceiling, but dequant-on-the-critical-path at M=1 will sit below it.
End-to-end B=1 tok/s and an MMLU parity check are the open bench rows.
