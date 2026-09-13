# Sparse decode ms/tick on the cold tier at 32k; 256k row pending-remote — H20, 2026-09-13

> Status: pending-remote (256k row; the 32k row is measured)

## Context

The sparse cold-tier wins so far (#556/#562/#565) are prefill-only. The
open defect
[2026-09-13-served-sparse-decode-eager-not-graph](../errors/2026-09-13-served-sparse-decode-eager-not-graph.md)
is a DECODE cost: a served sparse engine ran every pure-decode tick eager
(full candidate re-score + promote) because device selection was never enabled
by the serve path. This is the measured cost at B=1, k=128, on top of a
prefill that drove the host cold tier.

## What worked (measurement)

`scripts/trace_256k_decode_cost.py`: production sparse engine
(`sparse_k=128`, bounds, `--kv-cold-bytes` 6 GiB + 12 GiB mmap spill), one
B=1 row, CUDA-synced wall around every decode `step()`, first 8 decode ticks
excluded as warm-up, RSS sampled every second.

| ctx | prefill s | decode ms/tick median | p10 | p90 | timed ticks | RSS GiB (decode) | cold host GiB | cold SSD GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 25.5 | **88.23** | 82.63 | 97.64 | 55 | 12.12 → 12.32 | 1.147 shared | 0.0 |
| 262144 | pending-remote | pending-remote | | | 55 planned | pending | pending | pending |

H20 card 3, head `d488a22e` (#567 fused routing, pre-merge — single commit
based on b6b20bb3, the routing that merged), checkpoint
`/work/tilerl-ckpt/Qwen3.8-27B-NVFP4`. Raw artifact: `/work/dec32k.json`.

The median is the eager full-reselection cost named by the errors entry. The
p10–p90 band (82.6–97.6) is narrow, so the cost is steady-state, not outliers
(max 633.8 ms is the first warm-up tick and is excluded). Whole-run cold churn
(cumulative prefill+decode) was 11782 demotions / 9677 promotions; the harness
does not split decode-only churn. RSS is flat across decode and the shared
prefix stays pinned in host RAM at 32k (nothing re-spills: 1.147 GiB shared,
0 B on SSD).

The reference the errors entry contrasts against is the dense captured
decode tick at ~13 ms/tick (B=1, H20): sparse eager decode is ~6.8x that.
The 256k row did not run: the card ledger currently records all eight H20s
as aupai's under a 2026-09-13 04:0xZ user order
(`card_assignment.json`), so no tileRL launch is permitted without a named
lend. It needs one 256k prefill (359.5 s at #562) plus the same 64 decode
steps, same script with `--ctx 262144`.

## Rule

A prefill-only win does not price serving: the decode tick pays for
full-candidate re-selection every tick until the sparse graph + device
selection path of #557 is shown replaying under the serve default on a card.

## Results

| date | commit | machine | target | model | prefill s | decode ms/tick | note |
|---|---|---|---|---|---:|---:|---|
| 2026-09-13 | d488a22e | H20 card 3 | sm90 | 27B-nvfp4 | 25.5 | 88.23 | B=1, k=128, 32k, eager decode (pre-#557 serve path) |
