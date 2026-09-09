# The recorded sha was main's tip, not the tree that produced the number

**Date:** 2026-09-09
**Arch:** — (provenance, not performance)

## Context

`docs/experience/wins/2026-09-03-batched-selector-walk.md` records the B=1 W=8 spec arm
at **135.5 tok/s** (base 78.4, 1.728x) with sha `40bc83c`. The README quoted that
headline until 2026-09-09, when a reproduction on the current sha measured 126.5 and the
number was taken down (#354). To locate the 6.6% gap, the next step was measuring the
tick at the recorded sha.

## Root Cause

The recorded sha cannot produce the recorded rows. At `40bc83c`,
`scripts/acc_spec_arms.py` is hardwired to B=8 — `num_slots=8`, the MMLU and GSM8K
concurrency both the literal `8`, no `--concurrency` flag, and `arm()` has no
concurrency parameter. The entry itself states "`--concurrency` now exists; the B=1 rows
above are the first B=1 spec measurement taken." B=1 support landed in #58 (branch
commit `f49e006`, based on `40bc83c`; merged to main as `09657c0` after the walk squash
`a14a0a8`). A B=1 run at `40bc83c` is impossible.

The sha field was hand-filled with main's tip at the time the entry was written, not the
tree the run actually executed. The ancestry is consistent with two branch runs: the
B=1 rows most likely from the #58 branch (`f49e006` = `40bc83c` + B=1, without the walk
change), and the B=8 rows — which the entry's walk analysis needs — from the walk branch
(`053c742`, hardwired to B=8). Both recorded under one sha neither branch sat on.

The true provenance of the 135.5 row is **undetermined; most likely `f49e006`**. It is
not known to have ever run on a main sha.

## Fix

The reproduction and the tick comparison run at `09657c0` — the first sha on main that
contains both the walk change and B=1 support, so the only variable between it and the
current sha is the regression under test. The 135.5 row's provenance stays recorded as
undetermined; the number stands in its dated entry and nowhere else.

## Rule

A bench record's commit field must be the sha of the tree that produced the number,
taken from `git rev-parse HEAD` at run time — never hand-filled, and never the tip of
main. A hand-filled sha points at a plausible-looking tree that cannot produce the
record, and no gate catches it until someone reproduces. The bench schema's `commit`
field should be machine-verifiable against the run's own tree.
