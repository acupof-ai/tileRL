# GDN decode fused kernel: 47.16 ms/tok (was 65.46) — cuda/sm90, 2026-08-24

> Status: Shipped

## Context

The torch-eager GDN reference (`reference.gdn_forward`) was the single
largest decode cost: 47.4% of slice decode (13.085 ms / 2 GDN layers),
because it runs ~384 tiny kernel launches per layer per tick (48 value heads
x ~8 einsums each, all in a Python head loop). The fix is the SOTA fused
decode kernel from the tilelang `feat/qwen36-gdn-megakernel` branch — one
launch per (value head, batch), conv1d + SiLU + q/k-norm + decay-first delta
recurrence + gated RMSNorm + z-gate all in shared memory.

## What Worked

- **Ported `qwen36_gdr_decode_fused.py`** (branch `feat/qwen36-gdn-megakernel`,
  commit 0fb99503, unmerged) to `kernels_mma.py:make_gdn_decode_fused`.
  Adapted: f32 IO (branch is bf16-IO); separate NewState/NewWindow outputs
  (branch mutates in place); time-major conv window; split Q/Key/Val inputs
  (branch cats qkv — split so the QD/VD constexprs appear standalone in
  buffer shapes, which tilelang's constexpr matcher requires).
- **Registered in the sm90 cell** (`backend.py`): `linear_attn_chunk`
  dispatches to the fused kernel when `q.shape[1] == 1` (decode) and the
  sm90 cell is active; prefill (T>1) and other arches keep the torch-eager
  reference.
- **Parity gate**: `test_gdn_decode_fused_parity` compares the fused kernel
  against `reference.gdn_forward` on tiny shapes (b=2, nkh=2, nvh=4, kd=16,
  vd=16). On CPU it's a tautology (backend resolves to the reference); on
  CUDA sm90 it's the real gate — passed.
- **Measured delta** (real 27B slice, 2 GDN layers, H20, JIT-free steady
  state, same-shape warmup, gen=8):

| arm | ms/tok | total ms |
|---|---:|---:|
| torch-eager GDN reference | 65.46 | 523.7 |
| fused GDN decode kernel | 47.16 | 377.3 |

28% faster aggregate (1 prefill + 7 decode ticks). The fused kernel only
affects decode ticks: 146.4 ms saved over 7 decode ticks = ~20.9 ms/tick.

## Rule

A Python head loop over heads with per-head einsums is the default decode
bottleneck for GQA-style recurrent layers — one fused launch per (head,
batch) with the recurrence in shared memory is the SOTA fix, and the
tilelang constexpr matcher requires every `T.const` var to appear
standalone in at least one buffer shape (derived expressions like `NKH * K`
don't bind).

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | 762919b | H20 | cuda/sm90 | 27B slice (2 GDN layers) | — | 65.46 | 15.3 |
| 2026-08-24 | (this) | H20 | cuda/sm90 | 27B slice (2 GDN layers) | — | 47.16 | 21.2 |

ms/tok is the smoke-bench aggregate (1 prefill + 7 decode, gen=8); the
decode-only improvement is larger (~20.9 ms/tick saved). Prefill is
unchanged (T>1 still uses the torch-eager reference).

Raw artifacts: pod smoke bench stdout (H20, `/host/tc27-nvfp4-slice2`).
