# Native-fp8 in_proj_qkv+z fusion (qkvz) — sm90, 2026-08-26

> Status: Shipped

## Context

Each GDN layer projects the same post-norm hidden twice: `in_proj_qkv`
(h→10240 = qd+kd+vd) and `in_proj_z` (h→6144). In the Qwen3.8 checkpoint
both ship native-fp8 (e4m3 `.w8` + f32 per-128-block `.wscale`), so the
shipped GDN forward paid two `linear_fp8` launches — two activation quants
and two GEMMs on identical input. Projection fusion already existed for fp4
(`gate_up`/`qkv`/`ab`, decode win 2026-08-25); this extends the mechanism to
native-fp8.

Validity, verified in code: both projections take the same input tensor
(`model.py` `_gdn`), and both N dims are multiples of 128 (10240 = 80
blocks, 6144 = 48), so the per-128-block wscale concats losslessly along N
and the fused output splits back at `cfg.linear_qkv_dim`.

## What Worked

`_projection_groups` gains `{in_proj_qkv, in_proj_z} → qkvz`; `_fuse_projections`
is now format-aware — fp4 groups concat `.wq`/`.scale` as before, native-fp8
groups concat `.w8`/`.wscale` **and the bf16 master** (the CPU/decode path
computes with the master, unlike fp4 fused keys which have none), guarded by
`N % 128 == 0` on every member but the last. `_gdn` runs one `linear_fp8` on
the fused key and `autograd.slice`s the output at the qkv boundary. Same
serving-only gate (`fuse_projections=True`); training keeps the unfused
masters. The fused-key checks for `qkv`/`ab` were widened to `.w8 or .wq`
so the format-aware loader can never fuse a key the forward can't consume.

A/B at prefill shapes (M=512, K=2048, N=10240+6144), H20 pod, same process,
mean of 20 iters:

| arm | ms | rel-err |
|---|---:|---:|
| two-launch (shipped) | 0.2572 | 0.00e+00 |
| fused (qkvz) | 0.2204 | 0.00e+00 |

**1.17x** — bit-identical outputs (each output element is the same K-dot
product, so the fused GEMM is exact, not just close). The win is one fewer
launch plus one fewer per-token activation quant; the GEMM FLOPs are
unchanged.

Parity: fused vs unfused GDN logits allclose(rtol=1e-2) through the TileLang
CPU kernels (`tests/test_fused_projections_parity.py`, 128-aligned tiny GDN
dims — the wscale guard skips fusion for sub-128 N dims, which the real
checkpoint satisfies).

## Rule

Same-input native-fp8 projections with 128-aligned N concat losslessly —
fuse them; at prefill shapes expect ~15% on the pair (collapsed launch +
one fewer activation quant), with bit-identical outputs.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-26 | 4f862f9 | H20 pod | sm90 | microbench M=512 K=2048 N=10240+6144 | 0.2572 (pair) | — | — |
| 2026-08-26 | 30b040a | H20 pod | sm90 | microbench fused qkvz | 0.2204 (pair) | — | — |

Raw artifacts: `scripts/bench_fp8_qkvz.py` (pod, GPU 7).

## Iteration

Hypothesis -> verdict in 14.1 min agent wall time (1 pod round-trip, 10
edits) — one of two parallel A/Bs on GPUs 6/7 (workflow wall 14.1 min for
both, 236k subagent tokens). Single pod round-trip: the fusion, CPU parity
test, and bench script all passed on the first sync.
