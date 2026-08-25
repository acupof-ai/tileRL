# fp4 GEMV shared-memory dequant ping-pong rejected — 2.5-3.3x slower than register group4

> Status: Killed (correctness green, performance regression)

## Context

The grouped fp4 GEMV (`make_linear_fp4_gemv`, GROUP=4: load 4 micro-tiles,
decode all 4 into registers, FMA all 4) sits at ~46% of HBM roof on the big
projections (direct call, N=17408 K=5120, BW 3312); the nodecode floor is
~57%. The SGLang analysis (2026-08-25-sglang-fp4-kernel-comparison.md) pointed
at the remaining lever: dequant(k+1) overlapping FMA(k) — get the shuffle
issue slots off the FMA critical path. The register double-buffer (round 6)
spilled to local memory (22% roof), so this round tried **shared-memory
ping-pong**: dequant into a shared buffer, FMA from another, swap with a
barrier. H20 has 228KB shared — no register pressure.

Sweep (scripts/_sweep_gemv3.py, both shape orientations, direct call, all
variants bit-exact vs the shipped kernel — rel-err 2.74e-3/2.82e-3, maxdiff
0.000000):

| variant | N=17408 K=5120 | N=5120 K=17408 | vs group4 |
|---|---:|---:|---:|
| group4 (shipped) | 42.7% roof | 43.1% roof | 1.0x |
| group8 (GROUP=8 regs) | 41.8% | 42.8% | 0.98x (tie) |
| shared_pp (same-warp ping-pong) | 14.1% | 13.8% | 3.0-3.1x worse |
| producer (warp split, RING=3 SPSC) | 16.9% | 13.1% | 2.5-3.3x worse |

(Absolute %roof is depressed ~3pts by the co-tenant at 100% GPU — the shipped
kernel measured 46% uncontended. The ratios are the clean signal.)

Also checked: **bf16 IO is already done** — the shipped kernel takes X bf16
and writes Y bf16 (f32 accumulate); the MMA kernels' bf16-IO convention was
ported to the GEMV in round 1. No lever there.

## Root Cause

The shared-memory handoff adds a register→shared→register round-trip
(STS+LDS, 32 each per group) and a `bar.sync` per group, whose cost exceeds
the shuffle issue cost it was meant to remove.

1. **shared_pp (same-warp ping-pong):** the weights now travel
   shuffle→STS→barrier→LDS→FMA instead of shuffle→FMA. The 32 LDS land on
   the critical path in program order (the compiler emits all 32 LDS, then
   the 32 FMAs that consume them — LDS latency ~25 cyc is exposed, not
   hidden), and the 32 STS + 32 LDS on the load pipe plus the per-group
   barrier serialize the loop. The shuffle issue it removes (~32 cyc/group)
   is cheaper than the LDS+STS+barrier it adds. 3x worse.

2. **producer (producer/consumer warp split, threadIdx.z role, RING=3 SPSC
   ring, producer 2 groups ahead):** the consumer warp issues zero shuffles
   (the goal), but the ring handoff needs a `bar.sync` per group, and the
   consumer's prefetch LDS is pinned *after* the FMA in program order (the
   `if kg+1 < num_g` guard and the role branch stop the compiler from moving
   it before the FMA). So the LDS latency is exposed at each iteration start,
   the barrier forces producer/consumer lockstep (half the warps idle at any
   time), and the 48KB ring + 256 threads/block cuts occupancy
   (`__launch_bounds__(256,1)`). 2.5-3.3x worse.

3. **group8:** same instruction count per element as group4 (64 SHFL + 64
   FMA per group, same ratio), just more registers (ws[64] = 64 regs +
   Xs[64] bf16) → lower occupancy. Ties within noise, no benefit.

The grouped register decode is already at the local optimum: the 1-op/elem
shuffle LUT, hoisted 32 ahead of the FMA chain, hides its latency behind the
FMA dependency chain with zero extra traffic. Moving the dequant to shared
memory replaces a register-direct path with a shared-memory round-trip whose
latency (LDS ~25 cyc) is worse than the shuffle latency (~5 cyc) it removes,
and the barrier-per-group synchronization prevents any producer/consumer
overlap.

## Rule

For a memory-bound GEMV with a warp-shuffle LUT decode: keep the dequant in
registers and hoist the shuffles before the FMA chain (group4). Do not route
the dequant through shared memory — the STS+LDS round-trip and per-group
barrier cost more than the shuffle issue they remove. A register
double-buffer spills (round 6); a shared-memory ping-pong does not spill but
is 3x worse (this round). The shuffle LUT at 1 op/elem is the floor; the
remaining 46%→57% gap to the nodecode floor is not closeable by
reorganizing where the dequant lands — it needs fewer dequant instructions
per element (a narrower grid or a hardware decode path), not a different
buffer.

## Results

| date | machine | target | variant | ms (N=17408 K=5120) | %roof |
|---|---|---|---|---:|---:|
| 2026-08-25 | H20 | cuda/sm90 | group4 (shipped) | 0.0472 | 42.7 (contended) |
| 2026-08-25 | H20 | cuda/sm90 | shared_pp | 0.1429 | 14.1 |
| 2026-08-25 | H20 | cuda/sm90 | producer | 0.1194 | 16.9 |
| 2026-08-25 | H20 | cuda/sm90 | group8 | 0.0483 | 41.8 |

Raw artifacts: pod `/work/sweep7b.log` (both orientations, JIT-free),
`/work/sweep7.log` (first run). Sweep script: `scripts/_sweep_gemv3.py`
(diagnostic only — not shipped; the shipped kernel is unchanged).
