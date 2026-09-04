# SMEM staging is rejected — the sm70 GEMV is 1 flop per X byte, V100, 2026-09-03

> Status: **rejected before writing the kernel.** Task #26's surviving hypothesis
> (stage X in shared memory) is dead: the L1-capacity knee it predicts does not
> exist. The same arithmetic that kills it names the only remaining lever, and
> corrects the peak I have been measuring against for the last three entries.

## Context

Two candidates for the M=32 gap died at 1.01× (`n_partition`, splitting the FMA
accumulator chain). The one left was X load latency: X[32,5120] f16 is 328 KB, fits
the 6 MB L2 but not the 128 KB L1, so per-tile row reads were thought to hit L2 at
~200 cycles on the FMA critical path. ncu would settle it directly, but the pod
denies performance counters to non-root (ERR_NVGPUCTRPERM).

An A/B of a SMEM-staging kernel is a real restructure of a kernel with a documented
150× register cliff. A capacity limit makes a cheaper prediction first.

## The prediction

X's working set is `2·M·K` bytes, so an L1-capacity knee sits at

    M_knee = 131072 / (2·K)     →  M≈12.8 at K=5120,  M≈51.2 at K=1280

**A capacity limit's knee moves with K. A schedule limit's does not.** Sweeping M
at two K values separates them with no kernel change at all.

## Results

N=5120, `xh=True, sh=True`, np=4, block=32. `scripts/ab_gemv_l1_knee.py`.

| M | X KB | K=5120 µs | %FMA | X TB/s | | X KB | K=1280 µs | %FMA | X TB/s |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 10 | 71.8 | 2.3% | 0.73 | | 2 | 73.5 | 0.6% | 0.18 |
| 4 | 40 | 72.0 | 9.3% | 2.91 | | 10 | 65.1 | 2.6% | 0.81 |
| 8 | 80 | 90.1 | 14.9% | 4.66 | | 20 | 64.9 | 5.2% | 1.62 |
| **12** | 120 | 126.7 | **15.9%** | 4.97 | | 30 | 66.9 | 7.5% | 2.35 |
| 16 | 160 | 171.8 | 15.6% | 4.88 | | 40 | 74.9 | 8.9% | 2.80 |
| **24** | 240 | 241.7 | 16.6% | 5.21 | | 60 | 75.3 | **13.3%** | 4.18 |
| 32 | 320 | 304.7 | 17.6% | 5.51 | | 80 | 96.6 | 13.9% | 4.34 |
| 64 | 640 | 590.0 | 18.2% | 5.69 | | 160 | 192.2 | 13.9% | 4.36 |

**Both curves flatten at M≈12-24.** K=1280's X is 160 KB even at M=64 — it barely
crosses L1 at all — and it saturates by M=24. The predicted 4× shift in the knee
is absent.

## What that decides

**1. SMEM staging is rejected.** The mechanism it was built on (X spilling L1)
predicts a K-dependent knee, and there is none. Nothing was written.

**2. The peak I was measuring against is a factor 2 too high.** kernels_linear.py:638-651:
per row per tile the extern loads 2×`v4.u32` = **32 B of X** and issues 8
`fma.rn.f16x2` = **32 flops**. So the kernel is **1 flop per X byte, structurally**
— independent of M, `n_partition`, or cache behaviour. X traffic is `N·M·K·2` bytes
because every output column re-reads all of X, so the L1 port (128 B/cycle/SM × 80
× 1.53 GHz = 15.7 TB/s) caps this kernel at **15.7 TFLOPS = 50% of the 31.3 FMA
peak**, with a perfect hit rate.

The "4.1× off the FLOP floor, 24% of peak" in task #26 was measured against a peak
this kernel cannot reach by construction. **The real gap is ~2×, not 4.1×.**

**3. And L1 bandwidth is not the binding constraint either.** M=32 reads X at 5.51
TB/s = 35% of the L1 port. So all four candidates are now excluded: FMA issue
(measured, 1.01×), `n_partition` (measured, 1.01×), L1 capacity (this entry), L1
bandwidth (35%).

**4. The one lever the arithmetic leaves is raising flops per X byte** — one thread
computing several output columns, so one X load feeds 2-4 FMA sets. That is not an
analogy; it is the same equation solved for the other variable: at `ncols=4`, X
traffic falls 4× and the ceiling rises from 15.7 to 62.8 TFLOPS (above FMA peak, so
FMA becomes binding again, which is where a kernel should be). The register cost
lands on the decoded weights `d0/d1` (+8 per extra column), not on X, so the
150× cliff at kernels_linear.py:677 — which was about unrolling *M* — is not the
same exposure.

## The instrument, and why no number from its first run appears above

The first run of this script reported **5687 TFLOPS**, 182× V100's peak. `bk.timeit`
returns **milliseconds**; I divided as if microseconds. A number 182× above a
hardware peak describes the instrument, not the kernel, so none of that run's
derived values were reported — the timings were kept, the arithmetic redone.

The script now prints X TB/s beside TFLOPS and states both hardware ceilings in its
header, so the next impossible value is impossible to miss.

## Rule

**Prefer the prediction that costs nothing to test.** SMEM staging would have taken
a kernel restructure to falsify. Its mechanism implied a knee that moves with K, and
a two-K sweep of the shipped kernel falsified it in one pod run.

Second: **derive the kernel's own ceiling before calling a gap a gap.** Three
entries quoted "% of packed-f16 peak" for a kernel whose load/FMA ratio caps it at
half that peak. The ratio is two numbers in the extern — 32 bytes in, 32 flops out —
and reading them off first would have halved the reported gap and pointed at
flops-per-byte instead of at latency.

## Gate

None — no runtime change. `scripts/ab_gemv_l1_knee.py` is the reusable
discriminator: a candidate whose mechanism is capacity must move the knee with K.
