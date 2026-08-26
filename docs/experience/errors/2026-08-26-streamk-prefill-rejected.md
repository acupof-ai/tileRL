# Stream-K tile scheduling for fp4 prefill GEMM — REJECTED (0.549x geo-mean, occupancy wall), 2026-08-26

## Context

The shipped sm90 prefill GEMM (`make_linear_fp4_fp8_mma`, k_split=2) wins on
the 1-2-wave shapes (down/qkv/z/out, +4..18% vs k_split=1) but regresses on
gate/up (14 waves, 0.974x — atomics are pure cost there). Stream-K (tile
scheduling that distributes K-tiles across idle SMs without a fixed split)
targets exactly the low-wave regime: `num_sm` blocks partition the flattened
K-iteration space of the first tiles (split tiles f32 atomic-add into a
zeroed output), then each block takes full tiles. Ported from
`examples/gemm_streamk/example_tilelang_gemm_streamk.py` @ tilelang main onto
the dequant+fp8-WGMMA body, scheduling params as runtime args (one kernel
for every shape). Gate: ship only if geo-mean B/A > 1.0 vs the shipped
k_split=2, relerr vs arm A ≤ 1e-2.

A/B (H20 pod, GPU 6 idle, JIT-warm, mean of 20 iters per arm, same process
— contention-independent ratio; both arms zero the output inside the timed
region, commit eb6600f):

| shape (M,K,N) | A: shipped split2 ms | B: stream-K ms | B/A | rel-err vs A |
|---|---:|---:|---:|---:|
| 512,5120,17408 (gate/up) | 0.4376 | 0.8539 | 0.512x | 4.44e-03 |
| 512,17408,5120 (down) | 0.4749 | 0.8217 | 0.578x | 8.56e-03 |
| 512,5120,10240 (qkv) | 0.2707 | 0.5064 | 0.534x | 4.01e-03 |
| 512,5120,6144 (z) | 0.1665 | 0.3087 | 0.539x | 4.35e-03 |
| 512,6144,5120 (out) | 0.1784 | 0.3065 | 0.582x | 4.28e-03 |

geo-mean B/A = 0.549x — stream-K is ~1.8x slower at every shape. Correctness
is green (rel-err 4.0e-3..8.6e-3, the same split-reduction-order error as
k_split=2 vs k_split=1).

## Root Cause

Stream-K launches exactly `num_sm` = 78 blocks — one wave, one block per SM,
128 resident threads per SM (6% of Hopper's 2048). The dequant+WGMMA body is
occupancy-bound at these shapes: the XQ/WQ/WScale HBM loads and the WGMMA
wait need multiple resident blocks to hide latency, and the shipped split2
provides them (640 blocks -> ~5 blocks/SM, 640 threads/SM). Stream-K's
blocks run ~8x longer (down: ~1115 K-iters vs split2's 136) but that is not
the problem — the 1-block/SM occupancy is: the pipeline stalls on every
memory/WGMMA wait with nothing to switch to.

The scheduling family is wrong for the regime. Stream-K's target is the
under-filled grid (total tiles ≤ SMs), where 1 block/SM is the only way to
occupy the machine. The prefill shapes have 320-1088 tiles = 4-14 waves:
the grid is already over-subscribed, so stream-K's wave-packing buys
nothing, and collapsing to 1 wave costs ~2x. A re-tuned stream-K with more
programs (e.g. 4x SMs) would at best tie split2 — split2 already has
perfect load balance (640 blocks >> 78 SMs), and stream-K's only edge there
is fewer atomic adds, worth ~3% of kernel time against a ~2x occupancy
cost. No iteration flips this.

## Fix

None — reverted. The `make_linear_fp4_fp8_streamk` factory and its
`streamk_params` host planner stay out of `kernels_linear.py`; `registry.py`
was never wired. k_split=2 remains the sm90 default.

## Rule

Stream-K is for under-filled grids (tiles ≤ SMs): it packs idle SMs by
construction. On over-subscribed grids (4-14 waves) it collapses occupancy
to 1 block/SM and regresses ~2x on an occupancy-bound body. For the prefill
GEMMs the right lever is the one already shipped: fixed k_split=2, which
adds blocks (occupancy) instead of removing them. Do not re-run this A/B
unless a prefill shape appears whose tile count is at or below the SM count.

## Results

| date | commit | machine | target | arm | geo-mean B/A |
|---|---|---|---|---|---:|
| 2026-08-26 | eb6600f | H20 pod GPU 6 | cuda/sm90 | stream-K vs shipped split2 | 0.549x (reject) |

## Iteration

Hypothesis -> verdict in ~12 min agent wall time (1 pod round-trip) — first
pod run green (correctness OK, perf verdict immediate). Worktree fast-forwarded
to main first (branch forked before the split2 merge).
