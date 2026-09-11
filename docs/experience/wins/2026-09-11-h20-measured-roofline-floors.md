# The H20's measured roofline floors: ~3,314 GB/s and ~136.5 TFLOP/s — sm90, 2026-09-11

> Status: **Shipped** — two cards measured; the first calibration rows in the bench
> ledger, keyed `NVIDIA H20`, so `bench --kernels` now divides by measured floors
> instead of printing pending-remote.

## Context

`kernel_cost`'s roofline bound is `max(bytes / hbm_bw, flops / bf16_peak)`, and both
floors must be measured on the card that runs the tick — never a datasheet number.
The mechanism shipped 2026-09-11 (`2026-09-11-kernel-roofline-measured-floor.md`)
with the two rows pending until a card returned. The recall lifted 2026-09-11; cards
2 and 3 of an 8× H20 node were idle and calibrated first, because every later
runbook step (residency, kernel roofline, sparse-KV bounds) reads these rows.

Two pod facts the next card sitting needs (each wasted one run here):

- **Do not use `uv run` on this pod.** A fresh `.venv` resolves
  torch 2.11.0+cu129 in a build the container driver rejects
  ("The NVIDIA driver on your system is too old"), so `torch.cuda` is
  unavailable and every cuda-only command refuses. Run the system interpreter
  with the tree on PYTHONPATH (system torch 2.11.0+cu129 + tilelang 0.1.14
  work):
  `PYTHONPATH=/work/<dir>/src:/work/<dir>/packages/tilerl-kernels/src python3 -m tilerl.cli …`.
- **A clean synced tarball needs a `.synced_dirty` marker.** `pod_sync`
  stamped it only when dirty and removed it when clean, while
  `benchrec.git_dirty()` deliberately raises on a missing marker (a git-less
  tarball cannot self-check, and an absent marker must not mean clean), so the
  first clean calibration crashed before appending. Until the `pod_sync` fix in
  this PR merges, run `printf 0 > /work/<dir>/.synced_dirty` after a clean
  sync; afterwards sync writes `0` itself.

## What worked

`tilerl bench --calibrate --card 0` with exactly one card visible
(`CUDA_VISIBLE_DEVICES=N` makes the in-process index 0) appends two
`measured-best` rows per card: `hbm_bw_gbs` from a ≥1 GiB D2D copy
(read+write, CUDA-event median of 20) and `bf16_peak_tflops` from one 8192²
bf16 GEMM (2n³ flops). Both cards agree within noise, so the floor is the card
class, not one lucky die:

| physical card | hbm_bw_gbs (GB/s) | bf16_peak_tflops (TFLOP/s) |
|---:|---:|---:|
| 2 | 3316.31 | 136.54 |
| 3 | 3311.40 | 136.51 |

The pair differs by 0.15% on bandwidth and 0.02% on compute. Roofline steps
3/4 divide by these rows keyed on the exact device name `NVIDIA H20`.

Rows: `8630b2e321d5`, `90ff7463d4d0` (card 2), `db071a20813b`, `51339597c32f`
(card 3) in `docs/experience/bench/measurements.jsonl`, commit
`a08e546ad836246de272d599e557d56d0aadbb87`, `dirty: false`, target sm90. The
row's `device.card` is 0 on both (one card visible per run); the physical card
is recorded here and in the row command context.

## The clean-tree sync bug this step hit

The first clean-tarball calibration raised
`git_dirty: no git repo and no .synced_dirty marker` before writing anything:
`pod_sync` stamped `.synced_dirty` only when the tree was dirty and *removed*
the file when clean, while `benchrec.git_dirty()` deliberately refuses to
guess clean from an absent marker on the git-less pod tarball. Only dirty
pod runs could ever record. Fix: the clean branch writes `0`, matching the
`1` the dirty branch writes (`scripts/pod_sync.sh`). The failed runs wrote
no rows; the four rows above are from the fixed sync.

## Rule

A measured roofline floor is a per-card-class pair (bandwidth from a big copy,
compute from a big GEMM, event-timed medians), and two cards agreeing within
~0.1% is the check that the number belongs to the class. Provenance markers
for a tarball must have a value for BOTH states — an absent clean marker is
indistinguishable from a forgotten stamp.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-11 | H20 node, physical cards 2,3 (container `sglang-test`, host `iv-yeozpb5g5cbw80bls64e`) | sm90 | bw 3316.31 / 3311.40 GB/s; bf16 peak 136.54 / 136.51 TFLOP/s; 4 rows appended |

Raw artifacts: `docs/experience/bench/measurements.jsonl` (ids above).
