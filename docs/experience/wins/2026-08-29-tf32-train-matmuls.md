# fp32 matmuls in the backward were not using the tensor cores — cuda(H20), 2026-08-29

> Status: measured, -10.8% on a 27B train step. One line.

## Context

Training ran at ~14% of H20's bf16 peak. The total number said nothing about
what to fix, so the step was profiled per kernel
(`scripts/profile_train.py`, 64 layers, 1x256, GPU 7).

Eight rows of the profile were fp32 GEMMs on SIMT cores:

| kernel | ms | % |
|---|---:|---:|
| `cutlass_80_simt_sgemm_256x128` | 96.60 | 7.2 |
| `sm80_xmma_gemm_f32f32_f32f32_f32` 128x128 | 82.67 | 6.2 |
| `sm80_xmma_gemm_f32f32_f32f32_f32` 32x32 (nn/nt/tn) | 69.47 | 5.2 |
| `gemmSN_TN` / `gemmSN_NN` f32 | 45.08 | 3.3 |
| `cutlass_80_simt_sgemm_128x256` | 14.31 | 1.1 |
| **total** | **308.1** | **23.0** |

`simt_sgemm` is fp32 arithmetic on the CUDA cores — the tensor cores are idle.

## Root Cause

`torch.backends.cuda.matmul.allow_tf32` defaults to **False** since torch 1.12,
and the string never appeared anywhere in this tree. Every `@` in the
torch-eager backward (`reference.py`: `linear_bwd`, `linear_frozen_bwd`, the
whole gated-delta backward chain) therefore took the slowest path the hardware
offers.

## What Worked

Two lines in `Backend.__init__`, CUDA only:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

| | before | after |
|---|---:|---:|
| GPU-busy, 1x256 | 1332.4 ms | **1189.0 ms** |
| fp32 GEMM total | 308.1 ms | **115.6 ms** |
| kernels | 143783 | 144017 |

`sm90_xmma_gemm_f32f32_tf32f32_f32` and `cutlass_80_tensorop_s1688gemm`
replaced the SIMT variants. **-143 ms, -10.8%.**

Note the arithmetic does not close: those GEMMs gave back 192 ms but the step
only dropped 143 ms. The estimate of "-20%" made before the run was wrong; the
measured -10.8% is the number.

TF32 keeps 10 mantissa bits. That is inside the rtol=1e-2 parity gate on a
model whose weights are 4-bit, but it is a real precision change, so it ships
with the CUDA gradcheck, not the CPU one — TF32 does not exist on the CPU
target and the local suite cannot see it.

## What Is Left

The remaining step is two things, and neither is a GEMM:

- `linear_fp4_bwd_kernel` — 168 launches, 2103.9 us each, **353.5 ms (29.7%)**
- ~90k elementwise/reduce launches at 2-4 us — **~347 ms (29.2%)**

## Rule

Read the profile's kernel NAMES, not just its times. `simt_sgemm` vs
`tensorop`/`xmma_tf32` in the name is an 8x hardware path difference that no
amount of scheduling work recovers, and a library default put us on the wrong
one. Grep the tree for the flag before assuming a framework default is sane.
