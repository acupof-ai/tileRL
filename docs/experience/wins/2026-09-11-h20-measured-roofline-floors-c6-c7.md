# H20 cards 6 and 7 measured roofline floors: ~3,283 GB/s and ~136.1 TFLOP/s — sm90, 2026-09-11

> Status: **Shipped** — two more cards calibrated after the 2/3 pair
> (#488); all four now carry `measured-best` rows in the bench ledger.

## Context

Steps 2–7 of the [pending-remote runbook](../PENDING-REMOTE-CARDS.md) divide
declared bytes/flops by per-card measured floors, so every card gets its own
calibration before a roofline is printed. Cards 2 and 3 landed in #488
(3316.31/3311.40 GB/s, 136.54/136.51 TFLOP/s); this is the 6/7 pair, run with
the same command and the same interpreter.

The interpreter that works on this pod is **`/work/tl013/bin/python`**
(torch 2.11.0+cu129, tilelang 0.1.13), invoked directly: a fresh `uv run`
builds torch 2.13 whose CUDA 13.0 runtime the driver (12.9) rejects, and the
container has no importable system torch. One physical card is bound with
`CUDA_VISIBLE_DEVICES`; the in-process index is then 0, which is the value
`--card` and the row carry — physical identity lives in the launch command
and this entry, not the row.

## What worked

`tilerl bench --calibrate --card 0` once per physical card appends a
`hbm_bw_gbs` row (≥1 GiB D2D copy, read+write, CUDA-event median of 20) and a
`bf16_peak_tflops` row (8192² bf16 GEMM, 2n³ flops):

| physical card | hbm_bw_gbs (GB/s) | bf16_peak_tflops (TFLOP/s) |
|---:|---:|---:|
| 6 | 3307.97 | 136.42 |
| 7 | 3257.71 | 135.87 |

The two pairs agree on compute within 0.5% (135.87–136.54). Bandwidth spreads
a little more — card 7 is 1.8% below card 2 — but all four sit around the
class's ~3.3 TB/s; roofline bounds divide by the row for the exact device
name (`NVIDIA H20`), so the spread is carried, not averaged away.

A first run on each card stamped `dirty: true`: the synced tree carried
uncommitted edits and the pre-#488 `pod_sync` removed the clean marker
instead of writing `0`, so the rows were pulled from the shared store and
discarded. The accepted rows are from a clean-main sync (commit
7dce8909) after `printf 0 > .synced_dirty`, per #488's interim instruction.

Row ids in `docs/experience/bench/measurements.jsonl`: card 6
`e31c7dee750d` / `b25675d618e8`, card 7 `dcbae3d5cec0` / `2f5a79dc91c6`;
40-hex commit 7dce8909, `dirty: false`, `floor.kind=measured-best`.

## Rule

A floor row is only as good as its provenance: the dirty-tree rows measured
the same numbers but named a tree the commit did not contain, so they were
deleted rather than kept. And the pod interpreter is `/work/tl013/bin/python`,
not `uv run` and not a bare `python3`.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-11 | H20 node, physical cards 6,7 (container `sglang-test`) | sm90 | bw 3307.97 / 3257.71 GB/s; bf16 peak 136.42 / 135.87 TFLOP/s; 4 rows appended |

Raw artifacts: `docs/experience/bench/measurements.jsonl` (ids above).
