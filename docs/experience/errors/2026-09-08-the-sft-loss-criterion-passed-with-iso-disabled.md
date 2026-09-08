# P3's SFT-loss exit criterion passed with ISO's whole 2D path disabled — 2026-09-08

**Status:** fixed (`tests/test_iso.py::test_iso_lowers_loss_on_tiny_model`).

## Context

P3's optimizer half has four exit criteria and all four were already in the tree. The
fourth, "SFT loss falls", was `test_iso_lowers_loss_on_tiny_model:85`:

```python
assert losses[-1] < losses[0] - 0.1, f"ISO did not learn: {losses}"
```

A bound against the run's own first step, with no control.

## Root cause

The tiny model has **10 one-dimensional parameters and 17 two-dimensional** ones, and
`ISO.step_one` returns early for anything not 2D (`iso.py:101`), handing it to the base
optimizer. So Adafactor alone updates 10 weights on every step, whatever ISO does.

Stubbing ISO's entire 2D path to `return`:

| arm | first | last | drop | passes `> 0.1` |
|---|---:|---:|---:|:--:|
| real ISO | 24.7820 | 0.0244 | 24.7576 | yes |
| **2D path stubbed** | 24.7820 | 19.4631 | **5.3189** | **yes** |

The mutant clears the bound by **53x**. The criterion was green for an ISO whose entire
reason to exist was disabled.

## The obvious control does not work either

The natural fix — run pure Adafactor for the same 8 steps and assert ISO's loss is lower —
is not stable. Over four data seeds:

| seed | ISO drop | Adafactor drop | ISO better |
|---:|---:|---:|:--:|
| 0 | 24.7703 | 24.7589 | yes |
| 1 | 22.3072 | 22.4500 | no |
| 2 | 23.7535 | 23.7573 | no |
| 3 | 25.2548 | 25.3339 | no |

ISO wins on 1 of 4. At 8 steps on a tiny model the two are indistinguishable in loss, so
that assertion would be flaky in both directions — and a flaky gate is worse than the
weak one it replaced.

## Fix

Assert what actually separates the two: whether the 2D weights moved.

```python
moved = [k for k, p in model.params.items()
         if p.dim() == 2 and not torch.equal(p, before[k])]
assert len(moved) == len(two_d)
assert len(opt._frames) == len(two_d)
```

Real ISO: 17/17 moved, 17 frames cached. Stubbed: **0/17 moved, 0 frames cached.** Both
mutants redden — the 2D early return, and `p.copy_(((uu * ss) @ vv.T)...)` replaced by
`pass` so the frames train but never write back.

## Rule

**A loss that falls is not evidence the component under test made it fall.** Ask which
parameters the loss could have come from, and whether the component owns them. Here 10 of
27 were enough, and they belonged to the base optimizer.

**When the natural control is too noisy to discriminate, do not weaken to a bound —
change the observable.** The loss comparison could not separate ISO from Adafactor at this
scale; "did the 2D weights move" separates them completely and is not a threshold at all.
A criterion phrased as an outcome ("loss falls") often has a mechanical counterpart
("these weights changed") that is exact where the outcome is statistical.

Same shape as the merge gate that passed with a specialist dropped: the degraded arm still
produced a plausible number, because something other than the thing under test was
carrying it.
