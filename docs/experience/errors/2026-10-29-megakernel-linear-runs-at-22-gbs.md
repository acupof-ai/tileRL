# The single-launch megakernel runs 9.2x slower: its linear phase holds 2.4% of bandwidth — V100, 2026-10-29

> Status: open — the occupancy fix landed (1066 → 297 ms); the megakernel is
> still 2.6x slower than the existing schedule, so the single-launch goal is
> not yet a win.

## Context

A persistent instruction kernel runs all 64 layers of Qwen3.8-27B inside one
cooperative grid, so one decode token is one launch instead of about 700.
Correctness is settled: 6 teacher-forced tokens at B=1 match the existing
schedule to 8.0e-3, an 8-token fused prefill is bit-identical to 8 serial
steps, and zeroing the recurrent state after token 0 moves the logits by 0.92.

Speed was not. B=1, context 512, 32 steps, both arms in one process and both
orders: baseline 117.5 / 115.9 ms per token, megakernel 1066.5 / 1066.4.

## Root Cause

Phase attribution swaps same-shape program rows, so the compiled kernel is
reused and only the work changes:

| variant | ms |
|---|---:|
| all | 1065 |
| all minus linear | 37 |
| linear only | 1033 |
| attention only | 31 |
| gdn only | 4.1 |
| barrier floor (1059 `sync_grid`, no work) | 2.1 |

The linear phase reads 22.57 GB of resident weights in 1.03 s, i.e. 21.8 GB/s
against a 900 GB/s peak. The grid is fixed at `backend._sms` = 80 blocks of
128 threads, so at most 320 warps are in flight where the card holds 5120.
Raising the grid to 160 blocks gives 612 ms (1.74x); 320 blocks refuses to
launch with `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`, so the register
footprint, not the block count, sets the ceiling.

## Fix

`MegaForward` now defaults to `blocks = sms * 8` and
`annotate_min_blocks_per_sm(8)`. The occupancy sweep trades GDN's 128-float
per-thread local array for registers, and the cooperative grid grows to 640
blocks with it. GDN is 0.4% of the time, so spilling it is free.

Same bench, both arms, both orders, after the fix:

| arm | ms per token |
|---|---:|
| baseline | 113.8 / 116.0 |
| megakernel | 296.6 / 296.6 |

3.6x faster than before, and the linear phase reaches 89 GB/s. Still 2.6x
slower than the existing schedule end to end, and logits agree to 1.4e-2.

Rows per warp (giving one warp four independent weight rows to hide latency)
does not help. At grid 640 / 8 blocks-per-SM, full kernel and linear-only:

| rows per warp | all ms | linear ms |
|---:|---:|---|
| 1 | 297 | 255 |
| 2 | 307 | 265 |
| 4 | 381 | 339 |
| 8 | 502 | 460 |

Each extra row adds registers to a kernel already register-capped for
occupancy, so the loss of resident warps outweighs the longer memory chain.
The knob was removed rather than left at 1.

## Rule

A cooperative grid is bounded by what stays resident, so every register the
heaviest op wants is taken from the hottest op's parallelism. Measure phases
before tuning a persistent kernel: here 97% of the time sat in one op that the
launch-count argument never touches.

## Evidence

Remote logs are `final-bench.log` next to this run's tree. Probes:
`scripts/bench_megakernel.py` for the two arms above,
`scripts/probe_megakernel_phases.py` for the attribution table,
`scripts/probe_megakernel_occupancy.py` for the grid/register sweep,
`scripts/probe_megakernel_blocks.py` for the 160-block 612 ms point,
`scripts/probe_megakernel_model.py` for the 6-token parity and its negative
control, `scripts/probe_megakernel_prefill.py` for the fused-prefill
comparison, `scripts/probe_megakernel_ops.py` for the fused FP4 linear error,
and `scripts/probe_megakernel_fp8.py` for the on-demand FP8 decode.
