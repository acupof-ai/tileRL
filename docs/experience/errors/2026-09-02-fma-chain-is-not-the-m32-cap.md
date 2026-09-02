# The FMA dependency chain is not the M=32 cap either — V100 (sm70), 2026-09-02

> Status: hypothesis REJECTED by measurement, kernel reverted. Second lever
> closed on task #26. The remaining candidate is shared-memory staging, and this
> run is what narrowed it to that.

## Context

The sm70 fp4 GEMV is at 83% of bandwidth peak at M=1 and ~24% of packed-f16 FLOP
peak at M=32. `n_partition` was already measured flat
(`errors/2026-09-02-npartition-is-not-the-m32-lever.md`), so the next candidate
came off the extern's source rather than a guess.

Reading `tl_fp4_gemv_tiles_f16_m_xh` (`kernels_linear.py:829-833`): the eight
`fma.rn.f16x2` per tile all read-modify-write the **same** register.

    unsigned a = 0u;
    for (int i = 0; i < 4; ++i) {
      asm("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(xw[i]),     "r"(d0[i]));
      asm("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(xw[4 + i]), "r"(d1[i]));
    }

That is a serial 8-deep chain. At ~6 cycles of FMA latency it retires 8 issue
slots in ~48 cycles — **17% of issue peak**, which matched the ~15% of issue peak
computed independently from the instruction mix. Two routes to the same number is
what made it look like the answer.

The fix is cheap and does not touch the register constraint that governs this
kernel: split into four independent partials and tree-reduce. That costs 3
registers per row **body**, not per M, so the documented no-unroll rule (which
exists because unrolling `m` multiplies live `xw[8]` sets, and cost 150× when
tried) is untouched.

ncu would have answered this directly, but the pod denies performance counters to
non-root (`ERR_NVGPUCTRPERM`), so the A/B *is* the measurement.

## Thresholds, committed before the data

Written down before the M=32 rows came back, because "1.01× is basically 1.15×"
is an easy thing to talk oneself into afterwards:

- `>1.15×` → chain was the cap, ship it.
- `1.00-1.15×` → partially latency-bound; not worth non-bit-exact numerics alone.
- `~1.00×` → refuted; the cap is X **load latency**, not issue slots.
- `<0.95×` → `a[4]` spilled to local memory; abandon.

## What Worked — nothing, and the number is unambiguous

| M | shipped (ms/pass) | split | gain |
|---:|---:|---:|---:|
| 1 | 21.5 | 21.3 | 1.01× |
| 8 | 73.2 | 72.9 | 1.00× |
| 32 | 270.6 | 268.4 | **1.01×** |

Per shape at M=32: 1.00× / 1.02× / 1.01× / 1.02× / 1.00× / 1.02×. relerr
5.3-5.8e-04 everywhere, flat in shape and M — nonzero as expected (reassociating
an f16 sum is not bit-exact) and 17× inside the 1e-2 gate, so numerics were never
the blocker. The change is simply worth nothing.

**M=1 and M=8 being flat is consistent with the diagnosis, not evidence against
it.** At M=1 the tile does one row per weight load, so W bytes dominate and no
amount of latency hiding helps a bandwidth-bound kernel. Only M=32 could
discriminate, which is why it was the row that mattered.

## Root cause of the wrong prediction

The chain is real and its 17% arithmetic is right. It is simply **not the binding
constraint** — something longer hides it.

By elimination, that something is the X load the FMAs depend on. Per tile per row
the extern issues 2 × `ld.global.nc.v4.u32` of X and then eight FMAs *on those
registers*. Four independent FMA chains cannot hide a load their own operands
come from.

And here is the number I got wrong earlier: **I checked X against L2 and never
against L1.** X[32,5120] f16 is 328 KB. Against a 6 MB L2 it fits, which is what
correctly refuted the "X re-read 28× from HBM" model. Against Volta's **128 KB
L1 per SM it does not fit** — so the re-reads miss L1 and hit L2 at ~200 cycles,
on the FMA critical path, 64 times per tile at M=32.

"Cache-resident" was doing too much work in that earlier conclusion. L2-resident
is not free.

## What this leaves

Shared-memory staging is now the only candidate with a mechanism behind it rather
than an analogy: stage the X tile in SMEM once per block (~28 cycles, no miss)
and let the per-row reads hit that. V100 has 96 KB of SMEM per SM and the tile
slice is small. It removes the loads instead of reordering the math, which is the
one thing this run showed does not help.

Note the prize is still bounded by the same 4.1×, and staging is a real
restructure of a kernel with a documented register cliff — so it needs the same
M=1/8/32 A/B plus `bench_ctx_decode.py`, not just a prefill number.

## Rule

**Two independent derivations agreeing does not make a hypothesis true — they can
share an assumption.** The instruction-mix estimate and the chain-latency
estimate both said ~15-17% of issue peak, and that agreement felt like
confirmation. Both were computed *assuming issue slots were the constraint*; they
agreed with each other and not with the machine.

Second: **when a cost model says "it's cached", name the level.** L1 and L2 differ
by ~7× in latency on Volta, and a working set that fits one and not the other
behaves completely differently. The earlier entry's L2 check was correct for the
question it asked (does this miss to HBM?) and silently wrong for the one that
mattered next (is this on the critical path?).

Third, on the pre-committed thresholds: they cost two minutes and they are why
this reads as a clean refutation instead of "1.01×, marginal, maybe ship it
anyway". Write the decision rule before the number arrives.

## Results

| date | commit | machine | target | M | shipped | split | gain | relerr |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2026-09-02 | 3133117 | V100 32GB | cuda sm70 | 1 | 21.5 | 21.3 | 1.01× | 5.4e-04 |
| 2026-09-02 | 3133117 | V100 32GB | cuda sm70 | 8 | 73.2 | 72.9 | 1.00× | 5.5e-04 |
| 2026-09-02 | 3133117 | V100 32GB | cuda sm70 | 32 | 270.6 | 268.4 | 1.01× | 5.5e-04 |

Raw artifact: `scripts/ab_gemv_variant.py` (kept — it is the harness the SMEM
attempt needs, and it already sweeps the three M values that matter). The
`split=True` extern and factory flag were reverted; only the script remains.
`scripts/ncu_gemv_m32.py` is kept too, so the next agent finds the
ERR_NVGPUCTRPERM wall documented rather than rediscovering it.
