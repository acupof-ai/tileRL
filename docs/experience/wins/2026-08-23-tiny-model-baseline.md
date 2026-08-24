# Tiny model baseline — cpu, 2026-08-23

> Status: Shipped

## Context

First end-to-end integration of the tileRL package: TileLang CPU kernels,
hand-written autograd tape, paged-KV + gated-delta engine, fp4 weights,
OpenAI-compatible server. The workload is the `tiny` config (2 layers,
hidden 64, vocab 320) on macOS arm64 with no GPU — CPU is the portable
default and CI path. The one wall-clock metric that matters: prefill and
decode ms/tok on the tiny model, establishing the baseline all future
optimizations are measured against.

## What Worked

- **One TileLang backend, target-neutral kernels.** All kernels compile and
  run on CPU (`target="c"`) with block-parallel schedules only; no warp or
  warp-memory specifics. GPU targets (cuda/rocm/metal) share the same kernel
  source — pending-host verification.
- **Hand-written reverse-mode autograd tape.** Structural ops (reshape,
  transpose, slice, add, mul, sigmoid) are first-class recorded ops; views
  go through autograd helpers so the `id()`-based grad chain never breaks.
  Tape gradcheck passes against central finite differences
  (`rtol=5e-2, atol=5e-4` — tilelang f32 accumulation order differs from
  the reference).
- **Gated-delta (GDN) as one monolithic backend op.** The full GDN layer
  (conv1d, delta gate, RMSNorm, SiLU, recurrence) is a single
  `linear_attn_chunk` call with kwargs; the backward is a monolithic
  torch-eager reference (gradchecked, 11 grads).
  `# ponytail: torch-eager backward, tilelang kernel when perf demands`
- **Dense training path.** `kv.dense=True` switches the model to
  `backend.attention` (dense causal GQA) instead of paged attention,
  avoiding pool-indirection grad issues. Training KV holds only
  `LinearStatePool` (no paged pool).
- **fp4 e2m1 packing.** `pack_fp4` / `unpack_fp4` in `ops/fp4.py`;
  low-nibble-first, per-16-block scale `block_max/6`, LUT
  `{0,.5,1,1.5,2,3,4,6}`. Roundtrip error < 1e-2 on small-magnitude
  bf16 weights.
- **Prefix cache publishes all block-aligned prefixes.** A 24-token query
  now hits the 16-token entry (previously only the full 32-token prefix was
  inserted, so shorter queries always missed).
- **Engine `_drain` accumulates across polls.** `poll()` clears the finished
  dict each call; accumulating `done.update(engine.poll())` prevents
  early-finishing requests from being lost.

## Rule

CPU is the portable default: all kernels compile and pass on `target="c"`
with block-parallel schedules; GPU targets are the same source, pending-host.
The autograd tape records structural ops as first-class entries — never let
a view bypass the tape, or the `id()`-based grad chain breaks silently.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-23 | uncommitted | macOS-arm64 | cpu | tiny | 56.98 | 60.03 | 17.6 / 16.7 |

Raw artifacts: `tilerl bench` output (prefill total 7292.9 ms, decode total
1920.8 ms, prompt_len=128, gen=32).
