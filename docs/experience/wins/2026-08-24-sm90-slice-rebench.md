# sm90 slice re-benchmark: 31.09 ms/tok decode + prefill-512 profile — H20, 2026-08-24

> Status: Shipped

## Context

Re-benchmark of the 2-layer NVFP4 slice (`/host/tc27-nvfp4-slice2`, both layers
GDN) on an idle H20 after the fused GDN decode kernel landed, with the first
prefill-512 measurement and full-model extrapolation vs the 80 tok/s decode /
3800 tok/s prefill targets. Same config as the MMA entry:
`scripts/real_ckpt_smoke.py /host/tc27-nvfp4-slice2 --layers 2 --gen 8`
(JIT-free after same-shape warmup; `--train-steps 0` — training is not part of
this measurement and does not touch the timed generate).

## What Worked

- **Decode 48.85 -> 31.09 ms/tok** (8-token average: 1 prefill [1,16] + 7
  decode). The fused GDN decode kernel's share: per tick, `linear_attn_chunk`
  is 0.425 ms (4.6%) vs the torch-eager reference's ~21 ms/tick in the
  pre-fusion era. The earlier 47.16 ms/tok figure
  (`2026-08-24-gdn-decode-fused.md`) was measured on a GPU with 19 GB of a
  co-tenant's memory in use; this run on idle GPU 3 gives the clean number.
- **Per-op decode profile** (`scripts/profile_slice.py`, avg of 10 ticks):
  GPU sum 9.295 ms, wall 9.949 ms, dispatch overhead 0.654 ms (20.4 us/op,
  32 ops/tick). `linear_fp4` 8.289 ms (89.2%) is the whole story; of that,
  lm_head is 2.421 ms (measured in-engine) and the 16 per-layer gemms are
  5.868 ms (2.934 ms/layer).
- **Prefill 512 tokens: 11.2635 ms/tok -> 89 tok/s** (slice).
  `linear_attn_chunk` is 5657 ms of the 5766 ms tick (98.1%) — prefill
  (T>1) still runs the torch-eager GDN reference (~384 tiny launches per
  layer in a Python head loop). The fused kernel only covers decode.
- **Extrapolation to 64 layers** (naive, lm_head counted per layer):
  decode 313.1 ms/tok (3.2 tok/s), prefill 360.4 ms/tok (3 tok/s).
  Corrected (lm_head is once-per-tick, not 64x): decode ~239 ms/tok
  (4.2 tok/s). Targets: 12.5 ms/tok (80 tok/s) decode, 0.263 ms/tok
  (3800 tok/s) prefill.
- **Measurement caveat**: a standalone `linear_fp4` call on lm_head
  (N=248320) takes 107 ms when it is the first compiled (shape, dtype) in
  the process, but 2.4 ms in-engine — tilelang eager JIT specializes on the
  first call's shapes, and the engine compiles per-layer gemms first.
  Benchmark kernels in engine call order; the standalone number is a JIT
  artifact, not a real cost.

## Rule

Decode extrapolations count once-per-tick ops (lm_head) once, not per layer;
and kernel timing is only trustworthy in engine call order — tilelang's eager
JIT bakes the first compiled shape into the schedule, so a standalone
first-call measurement can be 40x off the in-engine cost.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | 3b1327f | H20 | cuda/sm90 | 27B slice (2 GDN layers) | — | 48.85 | 20.5 decode |
| 2026-08-24 | 53c1398 | H20 | cuda/sm90 | 27B slice (2 GDN layers) | 11.26 (512-tok) | 31.09 | 32.2 decode / 89 prefill |

Decode ms/tok is the smoke-bench 8-token average (1 prefill + 7 decode); the
decode-only slice rate is 100.5 tok/s (9.949 ms/tick). Prefill is the
512-token measurement from `scripts/profile_slice.py`. Extrapolated full
model (corrected): ~239 ms/tok decode (4.2 tok/s) and 360.4 ms/tok prefill
(3 tok/s) vs 80/3800 targets. Caveat: the slice has 2 GDN layers and 0
full-attn layers; the 27B's 16 full-attn layers are unmeasured (GDN
per-layer cost used as the average).

What still stands between us and the targets:

- **Decode (19x off)**: per-layer fp4 gemms at M=1 pad to 16 WGMMA rows
  (15/16 over-compute) — ~188 ms of the ~239 ms extrapolated full-model
  tick. A GEMV kernel for the decode path and bf16 IO (2x WGMMA throughput)
  are the upgrades. Dispatch overhead alone (20.4 us/op x 962 ops/tick =
  19.6 ms) already exceeds the 12.5 ms target — fewer, fused launches
  required.
- **Prefill (1000x off)**: the torch-eager GDN reference is 98.1% of the
  tick. A fused/parallelized GDN prefill kernel is the single blocker;
  nothing else matters until it lands.

Raw artifacts: pod `/work/bench_smoke.log`, `/work/profile_slice.log`,
`/work/lm_head_diag2.log` (H20, GPU 3, JIT-free).
