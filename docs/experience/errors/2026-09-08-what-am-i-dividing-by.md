# "What am I dividing by?" — eight bad numbers, none of them a mistimed measurement — 2026-09-08

**Status:** fixed (the probes named below; the rule is the artifact).

## Context

One session produced eight numbers that were wrong, across five probes, in one day. Seven
were caught inside the session; one reached a peer's architecture survey before being
retracted.

| # | number | what was wrong |
|---|---|---|
| 1 | `_sample_commit` 89.5% of rollout | four *inclusive* timers summed as a flat partition |
| 2 | `unattributed = -27.210 s`, gate passed | remainder defined as `wall - sum(parts)`, then asserted to sum to `wall` |
| 3 | fp4 GEMM 3.7 GB/s | CPU model, so each call was a 133.8 MB PCIe copy |
| 4 | `keep _MGEMV=3` | a search for the largest M answered by a yes/no on one candidate |
| 5 | ladder 1.27x faster than GEMV | M=3 GEMV against M=4 ladder — two things changed |
| 6 | bound is 21.896 GB | that is one row of a three-row table; the other 1.441 GB is KV and GDN state |
| 7 | weight stream = 67% of forward | 21.890 GB divided by an fp4-only rate; 49% of those bytes take `linear_fp8` |
| 8 | fp8 1.65x fp4 utilization | arms 9.5x apart in size, and interleaved in one process |

## Root cause

**Not one of the eight is a mistimed measurement.** The `perf_counter` placement, the
warmup, the synchronize positions and the iteration counts were right every time — a
clock sweep later confirmed the timing method reproduces to 0.9% across separate
processes (832.8 / 825.5 / 826.2 GB/s on one shape).

Every one of them is a ratio whose **numerator or denominator was not the quantity the
conclusion needed**:

- #1, #2, #6 divided by the wrong total.
- #3, #5, #8 divided a real time by bytes from a different situation than the one claimed.
- #4 divided the decision space wrong — it partitioned "boundary ∈ {1}" against
  "boundary ∉ {1}" when the question partitioned over every M.
- #7 put two kernels' bytes over one kernel's rate.

So a review that checks the instrumentation finds none of them. The timing is the part
that was never wrong.

## This refutes the default performance-review checklist

The standard questions asked of a perf number — is the warmup long enough, where is the
synchronize, did the clock ramp, are the iterations sufficient — have a hit rate of
**0 of 8** here. Every one of those was already right. And one of the eight left the
session and was quoted in an architecture survey, so the checklist's miss rate is not
academic.

The checklist needs a line, and it belongs **above** the timing questions because it is
cheaper and catches more:

> **For every ratio, write one line for the numerator and one for the denominator: what
> quantity it is, and where it came from — read / derived / assumed.**

A number already carrying its provenance looks like `1144.7 GB/s = assumed bytes
(numel × itemsize, derived) / measured time (read)`, which states in the same breath that
it is not a measured bandwidth.

## Fix

Before writing any ratio, write one line for the numerator and one for the denominator:
what it is, and where it came from — **read** (measured this run) or **derived**
(computed from a model). Five seconds.

Applied retroactively, this catches #1, #3, #6, #7, #8 — five of eight — because each is
a one-line contradiction the moment both sides are named:

```
#7  numerator:   21.890 GB, all quantized weights            (read, per-key dump)
    denominator: 1144.7 GB/s, linear_fp4 at M=8              (read, one kernel)
                 ^ two paths over one path's rate
```

The remaining three need their own checks: #2 wants a gate that can fail (the negative
remainder was the alarm, turned into a row by a self-referential assert), #4 wants a sweep
where the criterion says "largest", and #5 wants one variable per comparison.

## The one that escaped

#7 is the only one that left the session, and the difference was not its size — it was
that a peer had already written it into a survey. **A wrong number's cost is set by how
far it travelled, not by how wrong it was.** #3 was off by 100x and cost nothing, because
it was absurd on its face and never quoted.

Corollary, learned the same day: a number's units and magnitude do not reveal whether it
was read or derived. A derived 3.14 GB and a measured 3.746 GB look equally like facts,
and two sessions independently derived byte counts from a block size that turned out to be
wrong. **The source has to be recorded when the number is written; it cannot be recovered
from the number.**

## Rule

**Write the numerator and the denominator down before you write the ratio.** Name each
one's quantity and its source. A ratio is two claims and an operation, and the operation
is the part least likely to be wrong.

**A "utilization" computed as assumed bytes over measured time is not a utilization.**
`numel * itemsize` is a model of what a kernel reads, not a measurement of it; if the
kernel's tiling reuses a weight, the shortfall is reported as lower efficiency. Say
"assumed bytes / measured time" in the same sentence as the number, every time.
