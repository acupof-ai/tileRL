# Fused single-launch rmsnorm regressed decode — cuda(H20), 2026-08-27

## Context

The decode tick is kernel-count-bound (~785 kernels/tick, see
`2026-08-27-decode-latency-bound-not-bandwidth.md`). rmsnorm is the largest
population (~168/64L, TWO launches each via split-K partial+apply). Hypothesis:
collapse decode rmsnorm to ONE launch (one block/row reduces all N, normalizes
in place) to cut ~168 launches.

## Root Cause

**Measured −20%: decode B=1 40.9 tok/s vs 51.2 baseline** (logits still PASS).
The single-block kernel's serial `for k in T.serial(N)` reduction over N=5120 in
one thread-block is far slower per call than the split-K pair's parallel
reduction. Because the kernel sits inside the captured CUDA graph, its extra
latency lands directly on the critical path. Fewer launches, but each launch got
much more expensive — net loss.

The launch-count model is right that there are too many kernels, but WRONG that
every kernel's cost is a fixed floor: the split-K exists precisely because a
1-block reduction of a 5120-wide row is slow. Killing the parallelism to save a
launch is a bad trade when the reduction is wide.

## Fix

Reverted (kernels.py / backend.py / registry.py back to HEAD). No half-state.

## Rule

Cutting a kernel launch only helps if the replacement kernel is not itself
slower by more than the ~launch cost (~a few µs). A split-K/2-launch op that
exists for reduction parallelism is NOT free to collapse — measure the single
kernel's own latency first. The right rmsnorm win is **fusion into an adjacent
op** (norm→GEMV prologue: the GEMV block already loads the row, add the
reduce+scale there) so the normalize costs ~nothing extra AND the launch is
gone — not a standalone 1-block norm. Next lever should be a true fusion
(eliminates a launch with no new serial work), or attack `add` (128 elementwise,
genuinely cheap to fold into an epilogue), not more standalone kernels.

## Follow-up (2026-08-28)

The fusion works when the reduction stays parallel: `make_rmsnorm_fused_bf16`
— one block per row, 256 threads each summing a strided K-slice, a
block-wide `tvm_thread_allreduce`, then normalize and write bf16. In-graph
2.1 µs per call vs 3.2 + 1.7 for the split-K pair; parity 1.7e-3 (bf16 out);
kernels per 8-layer B=1 tick 142 → 125. The rule stands: it was the serial
reduction that lost, not the single launch.
