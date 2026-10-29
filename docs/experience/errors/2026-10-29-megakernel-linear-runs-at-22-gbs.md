# The single-launch megakernel runs 9.2x slower: its linear phase holds 2.4% of bandwidth — V100, 2026-10-29

> Status: open

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

Not landed. The occupancy sweep (`scripts/probe_megakernel_occupancy.py`)
trades GDN's 128-float per-thread local array for registers via
`annotate_min_blocks_per_sm` and raises the grid with it. GDN is 0.4% of the
time, so spilling it is free if linear gains warps.

## Rule

A cooperative grid is bounded by what stays resident, so every register the
heaviest op wants is taken from the hottest op's parallelism. Measure phases
before tuning a persistent kernel: here 97% of the time sat in one op that the
launch-count argument never touches.
