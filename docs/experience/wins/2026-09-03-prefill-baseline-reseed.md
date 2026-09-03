# Prefill baseline reseeded on current main — H20, 2026-09-03

> Status: Shipped. `prefill/{len512,len2048,len8192}/sm90` move from f001ed8
> (2026-08-29) to 9e3836b (2026-09-03). No code change.

## Context

The three prefill rows in `bench-baseline.json` were seeded before the
chunkwise-WY default flip ([2026-09-03-gdn-prep-post](2026-09-03-gdn-prep-post.md),
`6a2bfc2`), so the gate was comparing today's build against numbers it beats by
20%. `_GATE` is 0.97, so a build 19.3% slower than main still PASSed.

## What Worked

`scripts/bench_harness.py --suite prefill --source /work/Qwen3.8-27B-NVFP4
--gpu 5`, run 8 consecutive times on `origin/main` at 9e3836b. Each row is the
harness's own median of three readings; the seeded value is the median of the 8
run medians.

| row | seeded tok/s | min | max | run-to-run spread, n=8 |
|---|---:|---:|---:|---:|
| `prefill/len512/sm90` | **2689.8** | 2683.7 | 2696.8 | **0.49%** |
| `prefill/len2048/sm90` | **2671.7** | 2661.3 | 2676.7 | **0.58%** |
| `prefill/len8192/sm90` | **2558.6** | 2553.7 | 2563.6 | **0.38%** |

The gate FAILs below 0.97x the baseline: 2170.7 tok/s before, 2609.1 now.
Slack against a real regression at len512: 19.3% before, 3.0% after.

## The 6.4-6.9% below the number the WY flip recorded, and what it is not

The WY entry recorded 2887.6 / 2852.5 / 2729.0 on a box where all eight cards
were at 0 MiB / 0%. This box is not that box: GPU 4 runs a 27B GRPO job, GPU 6
sits at 97-100%, GPU 7 holds 43 GiB, host loadavg 6.9-23.5 of 180 cpus.

That gap is the host, not the tree. `b05bb3d` — the merge that made WY the sm90
default, unchanged since — was synced to its own directory and bracketed against
9e3836b in the same window, two passes each, alternating:

| depth | b05bb3d | 9e3836b | main / WY-merge | 9e3836b vs the quiet-window WY row |
|---|---:|---:|---:|---:|
| 512 | 2694.2 | 2688.0 | 0.998x | 0.931x |
| 2048 | 2666.4 | 2670.6 | 1.002x | 0.936x |
| 8192 | 2561.0 | 2555.3 | 0.998x | 0.936x |

The 21 commits between that merge and 9e3836b, 10 of them touching
`src/tilerl` or the kernels, cost prefill 0.2%, inside the 0.5%
run-to-run spread. The same code that read 2887.6 on an empty box reads 2694.2
here, so the 6.7% is the box.

Host loadavg does not explain it on its own: within the 8-run set, loadavg 6.9
and 20.1 differ by 0.26%. Whatever the other three cards cost GPU 5 is not
visible in `getloadavg`, and no quiet window was available to isolate it — those
are other people's jobs.

The seed is therefore conservative. A later run on an empty box reads ~2890,
clears `_RAISE` and ratchets the row up on its own.

## Rule

Reseed against the same commit's own prior reading, not against the ratio you
expect. A bracket arm at the old commit in today's window separates "the tree
regressed" from "the box is busy" for the cost of two extra runs; without it,
a 6.6% host effect reads as a code regression.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-03 | 9e3836b | H20 GPU 5, tilelang 0.1.13, torch 2.11.0+cu129 | sm90 | Qwen3.8-27B-NVFP4 | 0.3718 (len512) | not measured | 2689.8 |
| 2026-09-03 | 9e3836b | same | sm90 | same | 0.3742 (len2048) | not measured | 2671.7 |
| 2026-09-03 | 9e3836b | same | sm90 | same | 0.3908 (len8192) | not measured | 2558.6 |

Host during every run: GPU 4 27B GRPO (41 GiB, 30-73%), GPU 6 (2.8-40 GiB,
97-100%), GPU 7 (43 GiB, 0-37%), GPUs 0/1/2/3 idle, GPU 5 exclusively this job.
8 runs for the seed, 4 more for the bracket.

Raw artifacts (pod, `/work/`): `reseed.log`, `reseed_run_1..8.json`, `ab.log`,
`ab_wy_{1,2}.json`, `ab_main_{1,2}.json`.
