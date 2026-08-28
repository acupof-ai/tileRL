# The 27B training step was launch-bound, not compute-bound — 28x, 2026-08-28

> Status: Shipped

## Context

The first 27B LoRA training rows measured **80 seconds per step** at 1x64
tokens. Two guesses were made and both were wrong, which is the point of this
entry:

1. "The frozen backward's `reference.dequant_fp4` dominates — an int64 nibble
   tensor and a `repeat_interleave`, ~1.5 GB of temporaries, 448 times a step."
   A TileLang dequant fused into the backward GEMM bought **1.18x** (80.1 ->
   67.6 s). Correct change, wrong cause.
2. "Then it must be the GDN layers — `gdn_chunk_fused` is already the most
   expensive kernel in prefill."

## What Worked

Profiling the step instead of reasoning about it
(`scripts/profile_train.py`, per-kernel over one `train_step`):

```
8 layers, 1x64: GPU-busy 1221.0 ms, 671123 kernels
  torch elementwise            185030   2.0 us   30.8%
  torch vectorized elementwise 188253   1.4 us   21.6%
  cuBLAS gemvx                 110592   2.1 us   18.8%
  Memcpy DtoD                  148012   1.1 us   13.5%
  torch reduce                  36905   3.1 us    9.4%
  linear_fp4_bwd (ours)            24 770.7 us    1.5%
```

**671,123 kernel launches for one step.** Every tileRL kernel is in the double
digits; 668K of the launches are 1-3 us torch micro-ops. `gdn_backward` ran a
Python double loop over (time step, value head) — 48 launches per step per
layer where one would do — and the step was bound on launch overhead, not on
any kernel.

Only the time step is sequential; value heads are independent. Both scans now
run one set of batched einsums per step, and the value-head gradients fold back
onto the key heads with a reshape-and-sum (`h -> h // rep` is contiguous, so
that IS the scatter-add).

| shape | before | after | |
|---|---:|---:|---:|
| 1x64 | 80114 ms / 0.8 tok/s | **2416 ms / 26.5 tok/s** | 28.0x |
| 1x128 | 157843 ms / 0.8 | **3924 ms / 32.6** | 34.1x |
| 1x256 | 312840 ms / 0.8 | **6696 ms / 38.2** | 40.0x |

Kernels per step: 671,123 -> 20,753. Peak memory unchanged (46.9 / 50.5 / 57.6 GB).

## Rule

Count launches before optimizing kernels. A reference implementation that loops
in Python over a head or channel axis is not "slow torch" — it is a different
complexity class, and no kernel work on the 1.5% row will ever reach it. The
profile costs one run; two wrong hypotheses cost a day.

Corollary: the ponytail marker `# torch-eager backward, tilelang kernel when
perf demands` hides this. Torch-eager is fine; torch-eager *with a Python loop
over a parallel axis* is not, and the marker does not distinguish them.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-28 | see bench-baseline.json | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 (LoRA on frozen fp4) | — | — | **26.5 / 32.6 / 38.2** at 1x64/128/256 |

Raw artifacts: `/work/chain5.log` (the 671K profile), `/work/chain6.log` (after).
