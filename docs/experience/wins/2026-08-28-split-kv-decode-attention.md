# Split-KV decode attention with the GQA group as the M tile — cuda(H20), 2026-08-28

> Status: Shipped (sm90, pure-decode ticks S=1; mixed ticks keep the dense kernel)

## Context

B=1 decode fell from 87.5 tok/s at d512 to 28.7 at 32k: ~24 ms of the 36 ms
tick in 16 attention layers for ~4.3 GB of KV — 5% of bandwidth. The dense
MMA kernel launched one 64-thread block per (query head, row): 32 blocks on
132 SMs, each scanning the whole KV serially with synchronous gathers, and
the 4 GQA query heads of a group each re-read the same KV head.

## What Worked

`make_paged_attention_decode`: grid (KVSPLIT=16, Hkv, B). A block owns one
KV head and one slice of the KV length; the group's 4 query heads are rows of
the 16-row tile (rows ≥ G masked) so the slice is read once per group; it
emits its online-softmax partial (O, m, l) into a static workspace (one per
batch bucket, graph-capturable). `make_paged_attention_combine` merges the
16 partials per (b, kv head, g) in the scaled-log2 domain (empty slices carry
m = −inf, l = 0). Parallelism 32 → 2048 blocks at B=1... KV bytes /4.

Kernel-level (B=2, 32k, `scripts/parity_attn_decode.py`): relerr 3.5e-3 vs
the dense kernel (bf16 output rounding), **1.445 → 0.217 ms (6.7×)**.
In-graph at d512, B=1: decode 5.6 µs + combine 3.6 µs per layer vs 30 µs for
the dense kernel. The combine had to be a plain warp-per-row kernel
(lanes over d, splits unrolled, scalar locals): the first two versions — a
serial split loop, then a `T.Parallel(D)` body with a fragment per
iteration — ran at 40 and 66 µs per call and would have cost the B=1 tick
more than the dense attention at d512.

## 256K (later the same day)

`paged_attention_decode_64` (64 KV splits, selected when the pool reaches
past 64K tokens — host-static, so graph-safe) keeps each block's serial scan
≤ 4K tokens. Kernel-level at 256K, B=1: dense 22.0 ms → **1.52 ms per layer
(14.5×)**, relerr 2.6e-3.

End-to-end harness rows, B=1, one H20 (KV alone is 17 / 34 GB, so one
sequence per pool):

| depth | ms/tick | tok/s |
|---:|---:|---:|
| 131072 | 16.36 | **61.1** |
| 262144 | 20.81 | **48.1** |

256K decode at 48 tok/s on a single card is what makes 128K–256K RL rollouts
a budget question rather than an impossibility (docs/roadmap.md, P6).

## Rule

Decode attention needs parallelism along the KV length, not along query
heads; pack the GQA group into the M tile so the KV is read once. The
gathers are still synchronous (tilelang 0.1.13, no cp.async for elementwise
copies) — 16 splits hide that. Per-page `T.copy` (4 × 16-row copies per
64-token tile, 2-stage pipeline) was measured: correct but 0.630 ms vs
0.256 — the bulk-copy path costs more at page granularity than it hides.
Stays on the gather.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-28 | see bench-baseline.json | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | 0.55 | 11.0 (B=1, d512) | B=1 **90.9 / 87.3 / 87.9 / 78.6** at 512/2k/8k/32k (was 87.5 / 79.8 / 58.5 / 28.7); B=8 agg 308.6 |
