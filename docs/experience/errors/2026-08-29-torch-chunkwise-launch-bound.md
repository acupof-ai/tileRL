# cuBLAS does not already win the GDN prefill — the torch chunked form is launch-bound

## Context

`gdn_chunk_fused` scans T tokens serially and is **32.1% of a real 64-layer
prefill** (192 launches x 1663.9 us = 319.5 ms of 995.1 ms; 2058 tok/s
GPU-bound; sglang's B=1 fp8 prefill is 4022, and the 4908 quoted elsewhere is its B=8). The chunkwise-WY decomposition is all batched
matmul plus a triangular solve, so the cheap hypothesis was that torch/cuBLAS
already beats the serial kernel and no tilelang work is needed.

## Root Cause

It does not. A/B on the real model (H20, GPU 7, prompt 2048,
`TILERL_GDN_CHUNKWISE`):

| chunk | GPU-busy | kernels | tok/s |
|---|---:|---:|---:|
| shipped kernel | **995.1 ms** | 11,834 | **2058** |
| 32 | 1185.4 ms | 151,802 | 1728 |
| 64 | 1059.6 ms | 87,290 | 1933 |
| 128 | 1032.7 ms | 58,106 | 1983 |

The Python loop issues ~15 ops per chunk per layer, so kernel count goes up
5-13x and the path is launch-bound, not math-bound.

## Fix

The negative result is informative in the other direction: chunk=128 nearly
matches the serial kernel **while paying 58,106 launches**. The arithmetic is
therefore very cheap — the state update, 128x128 per token in the serial form,
happens once per chunk in the chunked one. Put the same math behind few
launches and it wins by a lot.

So: not the six-kernel upstream pipeline (which also needs a `solve_tril` the
tilelang examples do not ship, and would multiply launches per layer by 7).
Rewrite the INNER LOOP of the one existing kernel — same single launch, same
I/O contract, same conv/norm/gate fusion, same `keep_steps` — scanning T/C
chunks instead of T tokens.

## Rule

Measure the lazy version before writing the fast one, and read a negative
result for what it says about the fast one. "cuBLAS loses at 58k launches" and
"the math is cheap" are the same measurement.
