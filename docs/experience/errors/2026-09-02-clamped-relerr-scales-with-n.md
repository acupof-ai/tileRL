# A clamped relerr denominator scales with N and fakes a dtype bug — 2026-09-02

## Context

The f16-block-scale A/B compared the sm70 fp4 GEMV against its f32-scale twin and
printed, per shape:

| N | reported relerr |
|---:|---:|
| 1024 | 2.06e-01 |
| 12288 | 6.87e-01 |
| 248320 | 2.18e+01 |

A relerr of 21.8 means the output is 22× the reference — a kernel reading f32 out
of an f16 plane. That reading killed the same idea once already: an earlier
attempt was rejected with "relerr 2.0-21.3, implementation is wrong", the kernel
flag was reverted, and the whole line was written off as broken.

The kernel was correct both times. Measured against the output's own scale, the
error is **1.88e-04** and flat in N.

## Root cause

The metric, not the kernel:

```python
rel = ((y32 - y16).abs() / y32.abs().clamp(min=1e-3)).max().item()
```

`clamp(min=1e-3)` floors the denominator, so any output row that lands near zero
turns a tiny absolute difference into a huge ratio. It is a `max()` over N rows,
so the more rows, the likelier one lands there — the metric grows with N by
construction. The three "errors" above are one relative error of 1.9e-04 divided
by three different near-zero rows.

The tell was in the numbers all along: a dtype misread corrupts *every* element,
so it cannot care how many rows there are. **A metric that scales with problem
size is measuring itself.**

Worse, `scripts/benchkit.py:36` already had the right one:

```python
def relerr(actual, ref):
    """Max abs error relative to ref's abs-max (0.0 on an all-zero ref)."""
```

The repo's own helper normalizes by the reference's abs-max — exactly the fix —
and had been there the whole time.

## Fix

`scripts/ab_scale_f16.py` calls `benchkit.relerr`. With the metric corrected the
f16 plane passed `allclose(rtol=1e-2)` at every shape and shipped: dense decode
35.3 → 37.6 tok/s at 4096 ctx, +6.5-7.5% at every context.

## Rule

Before believing a bad parity number, check whether it varies with something it
physically cannot depend on. Error from a wrong dtype, a wrong layout, or a wrong
index is a property of the arithmetic; it does not scale with N, batch size, or
iteration count. If it does, suspect the yardstick.

And grep for the helper first. Two rejections of a correct optimization cost more
than reading `benchkit.py` once — the second one had already been paid for by the
first.
