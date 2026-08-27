# Decode glue: per-call casts, oscale muls, state-pool round trips — cuda(H20), 2026-08-28

> Status: Shipped

## Context

After the twiddle GEMV the B=1 tick was 16.2 ms; the in-graph profile still
showed ~25% in torch glue kernels (copies / index / elementwise). CUDA events
carry no Python stack and torch.profiler's aten stacks came back empty, so
`scripts/profile_glue.py` attributes every torch call to its innermost tilerl
`file:line` with a `TorchFunctionMode` during one eager tick. Per 8 layers,
B=1:

| glue | calls | cause |
|---|---:|---|
| `to[f32->bf16 (1,1,5120)]` | 23 | rmsnorm wrote f32, the GEMV wants bf16 |
| `to[bf16->f32 (5120,)]` | 17 | **norm weights re-cast every call** |
| `to[bf16->f32]` small vectors | ~28 | GDN dt_bias / a_log / norm / conv re-cast every call |
| `to[f32->bf16 (1,1,6144/17408)]` | 16 | silu_mul / attention wrote f32 |
| `to[bf16<->f32 (1,48,128,128)]` | 12 | GDN state pool bf16, fused kernel f32-IO (1.5 MB each way) |
| `mul` in `_epilogue` | 39 | per-row `oscale` as a torch mul |
| residual `add` | 16 | (left for a GEMV-epilogue fold) |

## What Worked

- `Backend._const_f32`: cached f32 (optionally padded) casts of PARAMETERS,
  keyed by data_ptr with a weakref identity check and `_version` (the old
  embedding-table cache, generalized). Used for norm weights, GDN vectors,
  `wscale`, `oscale`. Never for activations.
- sm90 cell registers bf16-writing `rmsnorm_apply` / `silu_mul`; the
  attention gate multiplies in the output dtype.
- `oscale` folded into both GEMV kernels' epilogues (`OScale[n]`).
- GDN state pool f32 on cuda (+1.2 GiB at 16 slots), and the fused decode
  kernel updates it IN PLACE at `[Slots[b], layer]` (`Backend.gdn_decode`,
  serving-only fast path; the tape-recorded gather -> linear_attn_chunk ->
  scatter path stays for training). Register-staged — see
  `errors/2026-08-28-gdn-inplace-raw-serialized.md` for the ×8.7 first cut.
- silu_mul bf16 in AND out on sm90.

Kernels per 8-layer tick 321 -> 192; GPU-busy 2.89 -> 2.10 ms (in-graph).

## Rule

Attribute glue at the Python level (`profile_glue.py`); the profiler's stacks
are not reliable for aten launches. A cast of a parameter inside an op is a
per-tick kernel — cache it or store the served dtype.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-28 | see bench-baseline.json | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | 0.55 | 13.35 (B=1, d512) | **74.9** B=1 (+21% vs 61.7); B=8 agg **212.6** (+15%); 69.0 @2k, 52.5 @8k, 27.1 @32k |
