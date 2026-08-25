# Projection fusion (qkv / ab / gate_up) — sm90, 2026-08-25

> Status: Shipped

## Context

Decode on the 27B slice is launch-bound: the small projections (GDN
in_proj_a/b at N=48, q/k/v at N=1024) sit at 0.1-3% of roofline — pure
fixed launch cost, not bandwidth. Same-input projections (same post-norm
hidden, all fp4-packed) were launched as separate GEMVs. Packed fp4 weights
concat losslessly along N (per-32-block scales are per-row), so each group
can be one GEMV with a view-split of the output.

Groups per layer: `{gate,up}` → `gate_up` (both layer types);
`{q,k,v}` → `qkv` (full-attn); `{in_proj_a,in_proj_b}` → `ab` (GDN).
Serving-only (`fuse_projections=True` in `load_hf`/`build_random`, on in
`cmd_serve`): training keeps the unfused bf16 masters — the fused key has
none, so its tape backward would have nowhere to land the STE grad. The
original packed tensors are deleted after concat (engine moves every param
to device, so dead copies waste HBM).

## What Worked

One GEMV per group + `autograd.slice` split (plain view without a tape).
A/B on slice4 (3 GDN + 1 FA layers), decode graph, avg of 30 ticks, same
checkpoint, same process count:

- decode: **1.821 → 1.734 ms/tick (+4.8%)**, 549 → 577 tok/s (slice)
- prefill: 0.0552 → 0.0545 ms/tok (+1.3%, within noise, directionally right)

Smaller than the eager-path projection (~40%): the graph path already
amortizes Python dispatch — the residual win is per-kernel device overhead
in the replay (9 kernels removed per slice tick ≈ 9.7 µs each). The eager
B>1 decode path (the decode graph is M=1-only) still pays full launch cost
per kernel, so fusion's value grows with batch decode.

Parity: fused vs unfused logits allclose(rtol=1e-2) through both
RefBackend and the TileLang CPU kernels (`test_fused_projections_parity`).

## Rule

Same-input same-dtype projections concat losslessly along N — fuse them;
on a CUDA-graph decode path expect single-digit-% (the launch cost is
already amortized), on eager paths (B>1 decode, prefill) expect more.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | (this) | H20 pod | sm90 | slice4 (4 layers) | 0.0552 | 1.821 | 549 (decode, slice) |
| 2026-08-25 | (this) | H20 pod | sm90 | slice4 fused | 0.0545 | 1.734 | 577 (decode, slice) |

Raw artifacts: `/work/fuse_base.log`, `/work/fuse_on.log` (pod).
