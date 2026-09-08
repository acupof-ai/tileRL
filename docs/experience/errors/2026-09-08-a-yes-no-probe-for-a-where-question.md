# A yes/no probe answering a where question — 2026-09-08

**Status:** fixed (the probe's criterion; `_MGEMV` itself is left at 3 deliberately,
see [wins/2026-08-29-m-row-gemv.md](../wins/2026-08-29-m-row-gemv.md)).

## Context

`_MGEMV=3` routes `2 <= M <= 3` to the M-row GEMV instead of the mma8 ladder. A
single-GEMM sweep suggested the boundary was wrong, so a two-arm probe timed both
kernels at M ∈ {1,2,3,4} by flipping `TILERL_MGEMV` at a fixed M.

The criterion was fixed in writing before the run: lower the boundary only if the ladder
wins at every M below it; one M where the GEMV still wins keeps the boundary.

The probe ran, the data was good, and it printed:

```
Ms in the current 2..3 range where the GEMV still wins: [2]
-> keep _MGEMV=3. The boundary is load-bearing at M=[2].
```

**`keep _MGEMV=3` was never measured.** The data underneath it says the boundary should
be 2.

## Root cause

The question is *where does the boundary belong* — a search for the largest M at which
the GEMV still wins. The probe asked *can the boundary drop to 1* — a yes/no test of one
candidate. Those are different questions, and the second one's "no" does not endorse the
status quo:

| M | GEMV ms | ladder ms | winner |
|---:|---:|---:|:--|
| 2 | 0.098 | 0.121 | GEMV |
| 3 | 0.154 | 0.122 | **ladder** |

M=2 keeps the GEMV alive, so the boundary cannot fall to 1 — which the probe reported
correctly. M=3 prefers the ladder, so the boundary should fall to 2 — which the probe had
the data for and never asked. The criterion given was per-M and correct; the
implementation folded it into a single boolean over the whole range, and then printed the
default as if the boolean had endorsed it.

## Why this one is harder to catch than the day's other instrument failures

Three earlier probes today measured the wrong thing: a flat partition over nested timers,
an assert that could not fail, a GEMM timing that was a PCIe copy. Each produced a number
that was false.

This probe measured the right thing and reported it accurately. **Every number it printed
was true.** What was false was the conclusion sentence, which named a value that was never
a candidate in the test. The output reads like a verdict, so nothing about it invites
checking — and the check that would catch it is not "is this number right" but "is this
the question I needed answered".

## Rule

**A search for an optimum cannot be answered by a yes/no probe.** If the criterion
contains *largest*, *smallest*, *first*, or *where*, the probe must sweep and report the
extremum it found. Testing one candidate and negating it leaves every other candidate
unmeasured, including the incumbent.

**A probe must not print the incumbent as a conclusion unless the incumbent was an arm.**
"Keep X" is a claim about X, and a run that never evaluated X cannot make it. Print what
was measured — "the ladder wins from M=3 up" — and let the reader apply the criterion.
