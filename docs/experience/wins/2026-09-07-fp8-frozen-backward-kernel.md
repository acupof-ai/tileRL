# The fp8 frozen backward gets a TileLang kernel: the scale is per tile, so it applies to the accumulator — H20 sm90, 2026-09-07

> Status: Shipped. `linear_fp8_bwd` (`kernels_linear.py`, factory `make_linear_fp8_bwd_mma`) is
> registered in the sm90 cell only and dispatched from `Backend.linear_frozen_bwd`
> (`backend.py:1140`) ahead of the torch-eager reference. Gates green on card 0. The op and
> step measurements are below, and the step gain is smaller than predicted.

## Context

The fp8 dX was the largest remaining eager op in the GRPO backward. That is what the fp4
backward warpgroup fix left behind (#240) — the same `linear_frozen_bwd` function served both
weight formats, and only the fp4 arm reached a kernel
([where-the-backward-goes](2026-09-07-where-the-backward-goes.md),
[fp4-backward-warpgroup](2026-09-07-fp4-backward-warpgroup.md)). The kernel computes
gx = grad @ W for a frozen fp8 weight and produces no weight gradient.

## What Worked

The whole of the port is one design point: where the dequant scale is applied.

The fp4 scale is one f32 per 16 elements of a weight row, so it varies inside a block and the
fp4 dequant macro applies it to the **operand** in shared memory before the gemm. The fp8 scale
is one f32 per 128x128 weight tile (`reference.dequant_fp8`), constant across both axes of the
block, so it applies to the **accumulator** after the gemm — the same shape as the existing fp8
forward `linear_fp8`. One consequence for the loop: the reduction over N steps by the scale
block (128), not by `_RED_TILE` (32), so one scale value stays valid for a whole inner gemm.

The grad is bf16 and the weight fp8, so the weight tile is widened to bf16 in shared memory
before the gemm. Quantizing the grad down instead would change what the tape computes, and that
is a precision decision the kernel does not get to make.

## Gates

`test_frozen_bwd_fp8_parity` — kernel against the eager oracle, M in 64/128/192, with and
without `oscale`, at n=k=256 so both axes span more than one scale block. 6 passed on card 0.

`test_frozen_bwd_fp8_gradcheck` — finite difference on the eager reference, deliberately on the
reference: it is the oracle the parity is measured against, so a wrong oracle would certify a
wrong kernel with both green.

Mutant verified on card 0: indexing `WScale[n, bx * block_N // block]` with `block_N` in place
of `block` — a scale misalignment that still produces finite, plausible numbers. It gave
`1 failed, 2 passed`. Local suite for the two touched files: 47 passed, 4 skipped.

## There is no executing CPU twin, and there cannot be one

tilelang's C backend has no sub-f32 type. Measured with three controls: `float8_e4m3fn` gives
`InternalError: Cannot convert type float8_e4m3fn to C type`, `bfloat16` gives the identical
error, and the same kernel at f32 compiles. That is why every kernel in `_CPU_KERNELS` is f32,
and it applies equally to the fp4 backward already shipped.

The eager `reference.linear_frozen_bwd` is the twin the hard gate asks for. The parity
comparison is real only on sm90; off it the test skips with that reason.

## Numbers

| quantity | before | after |
|---|---:|---:|
| `backward_secs` (`--no-instrument`, warm, C=128) | 23.194 | **22.264** |
| speedup on `backward_secs` | — | **1.042x** |
| the same saving on the STEP (85.617 s, `73433bb`) | — | **1.011x** |
| fp8 dX op row, instrumented warm step | — | 3.162 s / 14.39% / 1864 calls / **1.696 ms** |
| fp8 dX, bare, both arms one process (8 shapes) | 4.260 s/step | **3.018 s/step** |
| op speedup, measured | — | **1.411x** |

**1.042x on `backward_secs`, against 1.126x predicted, from a measured 1.411x on the op.** The
prediction assumed 2.68x on the op and that the bucket would take the whole saving. Both were
optimistic, and the entry keeps the predicted figures rather than deleting them.

The 0.930 s is **1.09% of a GRPO step** — 1.011x — because `backward_secs` is 26.1% of the
85.617 s step measured at `73433bb`
([the step is 74% rollout](2026-09-07-the-step-is-74-percent-rollout.md)). Every ratio above is
on the bucket; this is the only step-level number in this entry.

**The 133-139 TFLOP/s the prediction used is the bf16 GEMM floor, not the fp4 dX kernel's rate.**
The same run measures the untouched `linear_fp4_frozen` kernel at **67.0 and 69.6 TFLOP/s** — the
fp4 entry's own `vs_bf16` 1.96-2.05x against a 135-137 bf16 GEMM says the same thing. So an fp8
kernel at 51-77 is at **parity with the shipped fp4 dX**, and the prediction was asking for twice
the throughput any frozen-dX kernel on this card has reached.

The two arms are 23.194 (`wins/2026-09-07-fp4-backward-warpgroup.md`, warm, `--no-instrument`,
peak 63.8 GiB) and 22.264 (this branch on 2cce289, same flags, peak 62.62); the
`--frozen-shapes` arm ran later on 443df89, the same diff rebased, and is labelled by its own
sha rather than folded into the pair. Step 1 is not the comparison:
`prof_backward_ops.py` prints `step + 1`, so step 1 pays JIT — 44.533 here, and 47.962 on the
run whose registry compiled `linear_fp8_bwd` eight times.

**The op row cannot be divided by the published 4.298 s.** That figure was measured before the
fp4 warpgroup fix, on a tree where every op ran slower — the untouched `linear_fp4_frozen` row
reads 6.146 ms/call there against 2.378 here, so a 4.298 / 3.162 ratio would credit this change
with the other commit's clock. Dividing the step delta instead — 0.930 s over 1864 calls =
0.499 ms/call, eager 2.195 against 1.696, 1.29x — is an assumption wearing a measurement's
clothes: it presumes the whole step delta is this op, and the per-call figure came out of that
same delta, so restating the product proves nothing.

`bench_frozen_bwd` (`--frozen-shapes`) is the measurement: both arms, one process, one clock,
12 reps, on the 8 fp8 shapes the recorder found. Kernel **3.018 s/step** against eager
**4.260** — **1.411x**, per shape 1.218x (N=1024) to 1.492x (N=248320, the lm_head), all eight
between. So the 1.29x inference was **low**, and the step captured **74.9%** of the op's 1.241 s
saving, not all of it. Two independent confirmations that the arms are the same work: the
recorder counts 3728 calls over the two steps, exactly 2 x the instrumented 1864; and the bare
kernel's 3.018 s/step lands 4.6% under the instrumented row's 3.162, the gap being the
per-handler timer.

Both halves of the prediction were wrong, in the same direction: the op gained 1.411x where
2.68x was assumed, and the step kept 74.9% of that gain where 100% was assumed.

## What could still be wrong

**The step kept 0.930 of the op's 1.241 s. The missing 0.311 s is not attributed here.** The
bare bench runs each shape 12 times back to back on a resident model; inside a step the same
call arrives once, between other ops, so an allocator or cache effect the bench does not
reproduce is the likely place, but nothing here measures it.

**Whether 51-77 TFLOP/s is this kernel's ceiling is untested.** It matches the fp4 dX's measured
67-70 on the same card and in the same run, so it is not anomalously slow; but the fp4 kernel
reached that only after the warpgroup fix (#240) found 2.765x in a tile and a thread count, and
no equivalent sweep has been run on this one. `_snap_mma_tile(min(128, m), 128)` with `bN = 64`
and `thr = 128` were chosen to mirror the fp4 kernel's post-fix values, not measured here.

The bf16 widening of the weight tile costs shared-memory bandwidth, against the bandwidth the
accumulator-scaling saves by not writing a dequantized operand plane. Neither side of that
trade is measured separately here.

## Rule

A kernel-vs-eager ratio needs both arms in one process. Neither a published figure from an
earlier tree nor the step delta divided by the call count is that ratio — the first carries
another commit's clock, the second assumes its own conclusion. Here they read 2.68x (predicted),
1.29x (inferred) and **1.411x** (measured), and only the third is a number.

Read where a quantized format's scale varies before porting a kernel that uses it. A per-row
scale applies to the operand and a per-tile scale applies to the accumulator, and the second
also fixes the reduction step: the loop must step by the scale block so one scale value covers
a whole gemm. The fp4 backward is not a template for the fp8 one past that point.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-07 | pending | H20 pod (GPU 0) | cuda/sm90 | Qwen3.8-27B NVFP4, GRPO backward | — | — | — |

Backward-only change; no serving path touched. The metric is `backward_secs`: 23.194 -> 22.264,
1.042x, from 1.411x on the op. Raw artifacts on the pod: `/work/fp8op.log` + `/work/fp8_ops.json` and
`/work/fp8eager.log` + `/work/fp8_ab.json` (instrumented, two runs on 2cce289), the
`--no-instrument` arm that gives the 22.264, and `/work/fp8shapes.log` +
`/work/fp8_shapes.json` (`--frozen-shapes`, on 443df89).
