# fp4 GEMV is instruction-issue-bound, not bandwidth-bound — cuda(H20), 2026-08-27

> Status: verdict (ncu). Kills scale-dtype AND split-K for the fp4 GEMV;
> names the fix (twiddle decode + bf16x2 FMA, wins entry same date).

## Context

In-graph profile: GEMVs are 74% of the B=1 tick. fp4 GEMV on the 27B layer
shapes ran at 15–37% of the byte roofline (lm_head reaches 55%). Two cheap
byte/occupancy levers were tried first, then ncu.

## Root Cause

| lever | result |
|---|---|
| Scale f32 → bf16 → e4m3 (`ab_fp4_scale_dtype`, deleted) | gate_up 83.2 / 84.0 / 83.0 µs — **zero effect** |
| split-K ×2…×16 (`ab_fp4_ksplit`, deleted) | o_proj 40.9 → 41–42 µs at 1 280 → 10 240 blocks — **zero effect** |

ncu on gate_up (34816×5120): **DRAM 40%, L1/TEX 87%, SM issue 82% busy,
IPC 3.27**. 37.8M warp-instructions for 178M elements = **6.8 instr/elem**:
ALU 2.7 (nibble extraction), FMA 2.5 (FMA + bf16→f32 cvt + scale), LSU 1.4
(the warp-shuffle LUT is an LSU op). At 132 SMs × 4 × 1.82 GHz the issue
ceiling is ~5.5 T elem/s, the same order as the 4.3 T elem/s byte ceiling —
bytes and blocks are not what the kernel is waiting on.

Also: the eager A/B (`benchkit.ab`, cuda events around Python calls) floors
at ~40 µs of CPU launch overhead, so o_proj-sized kernels (~10 µs GPU) all
read "40 µs" regardless of kernel. Small shapes are only visible in-graph
(`scripts/profile_graph_kernels.py`) or under ncu.

## Fix

Cut instructions per element, not bytes: tilelang's twiddling decode (18 PTX
ops / 8 elems, packed bf16x2 out) + `fma.rn.bf16x2` (2 elems/op) → ~3
instr/elem. Shipped, see `wins/2026-08-27-fp4-gemv-twiddle-bf16x2.md`.

## Rule

A GEMV below ~50% roofline is not automatically bandwidth-bound. Run ncu
SpeedOfLight + pipe counters first: if issue-busy ≳ 80% and DRAM ≲ 50%, the
lever is instructions/element (decode width, packed math, tensor cores) and
every byte/occupancy experiment is wasted. Never A/B a <40 µs kernel with the
eager harness.
