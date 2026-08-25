# GDN chunk kernel: hoist NewState write out of the token loop — cuda/sm90, 2026-08-25

> Status: Shipped

## Context

The fused GDN chunk-prefill kernel carried the recurrent state column in a
per-thread local array (registers/L1) but still wrote the full 128-float
column to global `NewState` on **every token** (`kernels_mma.py:1111`, inside
the T-loop). The only consumer is `model.py:327`, which stores the chunk-end
state back to the state pool as the next chunk's seed — every intermediate
write was dead. At T=512, H=48 that is ~511 × 48 × 128 × 128 × 4 ≈ 1.6 GB of
dead traffic per layer.

## What Worked

Write `NewState` once, after the scan, from `state_local`. The per-token
store was the only global state traffic in the loop; the recurrence itself
was already register-resident. Decode rows (T=1) are unchanged — one write
either way. A/B in one process, same random inputs, slice4 prefill-512
shapes (B=1, T=512, nkh=16, nvh=48, K=V=128):

| arm | ms |
|---|---:|
| old (per-token write) | 1.993 |
| new (chunk-end write) | 1.699 |

14.8% kernel-level, `out` and `new_state` allclose (rtol=1e-3) — bit-identical
recurrence, only the dead stores removed. CPU parity green (GDN 3/3 tests).

## Rule

The recurrent state's global store is a chunk-end event, not a per-token one:
when the state lives in registers across the scan, every intermediate global
write is dead traffic — ~1.6 GB/layer at prefill-512 in this kernel.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | 7d750d7 | H20 pod | cuda/sm90 | Qwen3.6-27B NVFP4 slice4, GDN chunk kernel, T=512 | 1.699 (kernel) | — | — |

Raw artifacts: one-shot A/B `scripts/bench_gdn_writeback.py` (deleted after
verdict; numbers above are the record).
