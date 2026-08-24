# Decode tick CUDA graph capture: dispatch 18.3 ms -> 0.04 ms — H20/sm90, 2026-08-24

> Status: Shipped

## Context

Decode dispatch is 899 ops/tick x 20.4 us/op = ~18.3 ms on the full 27B
(extrapolated from the slice; the rebench measured 20.4 us/op on an idle
GPU). That alone exceeds the 12.5 ms (80 tok/s) target — before any GPU
work. The decode tick is static: the same ops, the same shapes, every
token, for the life of the process. design-engine.md says it is a captured
kernel sequence, not an interpreted one; this entry lands that invariant.

## What Worked

- **`_DecodeGraph` (engine.py)**: static input buffers (input_ids,
  positions, block_table, seq_len, state_slot) + the engine's own static
  KV/state pools, warmed up (tilelang JIT is host work — it must finish
  before capture) then captured once per batch-size bucket (day-1: M=1)
  with `torch.cuda.graph`. Per tick: copy the 5 inputs into pinned staging
  buffers, `non_blocking=True` H2D into the static buffers, `replay()`,
  read the static logits. Sampling stays outside (it syncs). Auto-on for
  CUDA; eager stays the default elsewhere and the fallback on capture
  failure (loud warning).
- **Prerequisite — `write_tokens` scatter kernel (kernels_mma.py)**: the
  pool's host loop did `int(kv.seq_len[bi])` / `int(kv.block_table[...])`
  per token — GPU->CPU syncs, uncapturable. One sm90 tilelang kernel
  (vLLM reshape_and_cache indexing: blk = block_table[b, pos // 16],
  off = pos % 16) replaces it; other arches keep the torch loop as the
  backend-op fallback. Also removes the per-token syncs from eager decode.
- **Prerequisite — `_inv_freq` on device**: the cached inv_freq was a CPU
  tensor, so every `rope` call did a CPU->CUDA `.to()` — illegal inside
  capture ("Cannot copy between CPU and CUDA tensors during CUDA graph
  capture unless the CPU tensor is pinned"). Cached on the backend device
  now; also kills a per-call H2D copy in eager.
- **Pinned async input copies**: the naive `copy_(torch.tensor(...))` from
  unpinned CPU tensors is synchronous — 7.3 ms/tick under GPU contention
  (measured). Pinned staging + `non_blocking=True` makes the copies async:
  36 us for all 5. Stream ordering keeps replay after them.
- **No tilelang bug**: tilelang 0.1.13 eager kernels capture and replay
  cleanly (smoke: one gemm capture + replay, bit-identical output). The
  capture blockers were all in tileRL's host code (syncs, CPU-cached
  tensors, synchronous copies), not the kernel runtime.

## Rule

The decode tick is replayed, not interpreted. Capture prerequisites:
(a) no host syncs inside the forward (the write_tokens loop had them),
(b) every cached tensor device-resident (inv_freq was not),
(c) per-tick inputs fed through pinned async copies. The replay itself is
3 us — the launch cost is independent of op count, so capture kills the
899-op dispatch tax in one shot. The remaining decode cost is GPU work
(the GEMV-era ~2 ms/tick slice GPU sum), not dispatch.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | e3b8d22 | H20 | cuda/sm90 | 27B slice (2 GDN layers) | — | 0.040 (captured dispatch) | — |

Dispatch (CPU-side, no sync, min of 50 ticks, 2 GDN layers; the pod had a
100%-utilization co-tenant, so eager is contended — captured is
contention-robust thanks to async copies):

| path | min ms/tick | breakdown |
|---|---:|---|
| eager forward | 12.154 | 32 kernel launches (idle-GPU extrapolation: 32 x 20.4 us = 0.65 ms) |
| captured run | 0.040 | pinned copies 0.036 + replay 0.003 |
| speedup | 304x | (idle-GPU: ~16x on the slice; ~460x full-model) |

Full-model extrapolation: eager 899 ops x 20.4 us = 18.3 ms dispatch ->
captured ~0.04 ms (copies + replay, op-count-independent). Target was
~1 ms; beaten 25x. GPU work is unchanged (same kernels replayed) — the
GEMV-era slice GPU sum (~2 ms/tick) is the remaining decode cost, not
dispatch.

Parity: `tests/test_decode_graph.py` (CUDA-only) — eager vs captured
token streams identical on the tiny model (greedy, same seed), with an
assertion that capture did not fall back to eager. `test_ops_parity.py`
26/26 on CUDA (including the new `write_tokens` kernel vs the torch loop).

Raw artifacts: pod `/work/dispatch_final.log`, `/work/parity_full.log`,
`/work/graph_parity3.log` (H20, GPU 1, JIT-cached).
