# n_partition is not the M=32 lever — V100 (sm70), 2026-09-02

> Status: hypothesis REJECTED by measurement, and it was already refutable from
> data on hand before the run. Task #26's lever (c) is closed; the kernel is
> issue-bound on per-row X reloads, not bandwidth-bound on X re-reads.

## Context

The sm70 fp4 GEMV is at 83% of bandwidth peak at M=1 and ~24% of packed-f16 FLOP
peak at M=32 — near-optimal in the regime it was tuned for, 4.1× off in the one
prefill uses. The kernel grids `ceildiv(N, n_partition)` blocks with
`n_partition=4`, and **each block reads all of X[M,K]**, so X is re-read N/4
times: 2.85 GB against 0.10 GB of weights for `gate_up`, 28.4×.

That framing made `n_partition` the obvious lever — raise it, cut the re-read
count proportionally, and it needs no kernel change because `n_partition` is a
runtime arg.

## The hypothesis was refutable before the run, from numbers already measured

The re-read model implies an HBM traffic floor. Applied to **decode** (M=1, where
the same kernel is already measured in-graph at 19.35 ms):

    W 14.41 GB + X re-reads 12.81 GB = 27.22 GB -> 30.2 ms at 900 GB/s
    measured: 19.35 ms

**Faster than its own floor**, which is impossible — the model implies 1.41 TB/s
on a 0.9 TB/s bus. So the re-reads do not come from HBM, and the reason is
immediate once looked at: X[32,5120] f16 is **328 KB against a 6 MB L2**. Every
re-read after the first is an L2 hit. The compulsory miss is 154 MB per pass, 1%
of the weight stream.

I ran the sweep anyway, because a cheap negative control on the pod is worth more
than my confidence in an argument I had just constructed.

## What the sweep says

Microbench at M=32, four shapes carrying 88% of the weight bytes:

| shape | np=4 | np=8 | np=16 | np=32 |
|---|---:|---:|---:|---:|
| gate_up 34816×5120 | **1981.2** | 2003.3 | 2050.6 | 2767.6 |
| down 5120×17408 | 973.8 | **930.4** | 952.5 | 1393.7 |
| qkvz 16384×5120 | **949.4** | 958.0 | 957.5 | 1372.8 |
| gdn out 5120×6144 | 356.6 | 354.3 | **342.9** | 396.5 |
| **weighted total** | **251.8** | 250.7 | 254.6 | 351.3 |

**Flat from 4 to 16 (1.00×, 1.00×, 0.99×) and 28% WORSE at 32.** Cutting X
re-reads 4× changes nothing, exactly as the L2 argument predicts. np=32 regresses
for an unrelated reason: threads = 32 × n_partition, so 32 gives 1024
threads/block and caps occupancy at one block per SM.

Outputs were asserted bit-identical across every np, so this is a pure schedule
sweep and not a numerics trade.

## Where the time actually goes

From the extern's own source (`kernels_linear.py:820-838`), per 16-element tile:

- **once per tile**: 1 × `ld.global.nc.v2.u32` (W) + 2 × `tl_fp4_decode8_f16`,
  ~20 instructions.
- **per row, all M of them**: 2 × `ld.global.nc.v4.u32` (X) + 8 ×
  `fma.rn.f16x2` + a pack + an `fmaf`, ~13 instructions.

At M=32 that is 32 × 13 + 20 = **436 instructions per 16 weight elements, 27.2
per element**, and **64 X loads per tile to feed 8 useful FMA pairs**. Against
V100's issue rate (80 SM × 4 schedulers × 1.53 GHz = 0.49 T instr/s) the kernel
is at ~15% of *issue* peak — a worse ratio than its 24% of FLOP peak, which is
the signature of spending issue slots on loads rather than math.

The design comment is explicit that this is deliberate: unrolling the M loop
"spills registers (32 bodies × ~25 regs >> 256/thread) and was 150× slower". So X
is deliberately reloaded per row to keep one row's worth live at a time. That
trade is right at M=8 and is what costs 4× at M=32.

**The lever is X residency across the tile loop, not the block geometry.** Either
hold several rows' X in registers and accept fewer tiles in flight, or stage X in
shared memory once per block so the per-row reads hit SMEM instead of L1/L2.
Neither is a parameter change; both are kernel work, and both must be measured
against the M≤8 rungs they share a factory with.

## Rule

**A traffic model must be checked against a measurement it did not come from,
and the cheapest check is whether it predicts something already known.** This
model was falsified in one line by the decode number that had been on record for
two days: it predicted a floor slower than the measured time. A model that says
the hardware cannot do what it is observably doing is wrong, and noticing that
costs nothing.

Second: **"re-read N times" is only a cost if it misses cache.** 328 KB against
6 MB of L2 makes the re-read count irrelevant, and the number that mattered (328
KB, the working set) was never in the calculation — only the multiplier was. Ask
what the working set is before multiplying.

Third, on running it anyway: the sweep cost one pod job and produced the negative
control plus an unrelated real finding (np=32's occupancy cliff). An argument that
a measurement is unnecessary is worth less than the measurement when the
measurement is cheap.

## Results

| date | commit | machine | target | shape set | np=4 | np=8 | np=16 | np=32 |
|---|---|---|---|---|---:|---:|---:|---:|
| 2026-09-02 | cb494f2 | V100 32GB | cuda sm70 | 88% of trunk fp4 bytes, M=32 | 251.8 | 250.7 | 254.6 | 351.3 |

Raw artifact: `scripts/ab_gemv_npartition.py`. It cross-checks bit-equality
across np and asserts N divides every np under test (the kernel writes
`Y[m, bx*np + ni]` with no `n < N` guard, so a non-dividing N would write past
the end rather than produce a wrong number).

Reconciliation: scaled to all 305 launches the microbench predicts 4558 ms per
512-row prefill against 3406 measured in-graph, 0.75×. Same order, so it is a
usable A/B harness at M=32 — unlike M=1, where its ~60 µs eager launch floor
dominates the small shapes.
