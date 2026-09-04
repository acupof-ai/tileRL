# The mma8 N rung does not order with the plan's, and one axis checked read as two

**Date:** 2026-09-04
**Arch:** sm90 (found), sm70 (same code path), cpu (gated)

## Context

`_pad2d` was given a guard that raises on a negative pad, replacing `F.pad`'s
silent crop. The fp8 mma8 branch broke immediately and was fixed at its own call
site. Publishing that fix, I wrote: *"the fp4 twin does NOT pre-pad x2, which is
why only fp8 breaks."* A peer's sm90 suite then failed both fp4 parity tests with
`ValueError: _pad2d: (64, 256) exceeds the target [32, 256]`.

## Root Cause

Two rungs, neither dominating:

```
Np   = _round_up(N, bN)     bN = the plan's N tile, 64 on the fp4 decode arm
Np32 = _round_up(N, 32)
```

`backend.py:581` pads `wq`/`scale` to `Np`; `:585` re-targets them to `Np32`.
`_round_up(n, 64) > _round_up(n, 32)` for n = 24, 40, 96, 130 — a class of N, not
one shape. `pack_fp4(torch.randn(24, 32))` in the parity test is N=24, so
Np=64 against Np32=32: the reported shape exactly.

My claim was true on the axis I checked and false overall. The fp4 branch does
not pre-pad on the **M** axis — it pre-pads on **N**. I compared the two branches
on the axis the fp8 bug happened to occupy, found nothing, and reported absence
everywhere. One axis checked, two axes present.

The peer's reading of the defect class was right and mine was not: the fp8 fix
was site-local, the reliance on `F.pad`'s crop was not. There were three sites,
found by arithmetic rather than by waiting for tests to trip over them.

## Fix

`_fit_rows(t, rows, cols)` drops rows off a plane already padded to a wider rung
and pads otherwise. Applied at all three re-target sites: fp4 `wq`/`scale`, fp8
`w8`/`wscale`, fp8 `x2` (replacing a hand-rolled slice). The other 10 `_pad2d`
call sites were checked individually: the two residual ones reshape to `(M, N)`
with `M <= _MX`, so their pads stay non-negative; the rest pad from unpadded
tensors.

The gate is the equality, not the shape: rows between the real N and either rung
are zero pad, so `_fit_rows(_pad2d(p, Np), Np32)` must equal `_pad2d(p, Np32)`
elementwise. `test_the_mma8_N_rung_can_be_narrower_than_the_plans_and_the_drop_is_exact`
asserts both that `Np > Np32` actually occurs (else `_fit_rows` is dead code) and
that equality — on the CPU target, where neither mma8 branch can execute.

Negative control, no card: `_pad2d: (64, 8) exceeds the target [32, 8]`.

`main` passes both fp4 parity tests **while feeding the kernel a `wq` cropped
from 64 rows to 32**. The cropped rows are zero, so its numbers were right by
luck. Second instance of the same class as the fp8 site.

## Rule

An asymmetry between two code paths is a claim about every axis they index, and
checking one axis licenses a statement about that axis only. Before writing "A
does not do X, which is why only B breaks", enumerate the axes X can happen on
and check each — or say which one was checked.

Corollary, from the peer: when a guard turns a silent behaviour into a hard
error, the sweep is every call site asked "can the first argument exceed the
target here", not the sites that tests happen to reach. Tests find the reached
ones; arithmetic finds the rest.
