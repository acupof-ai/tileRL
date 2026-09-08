# A hardcoded p=0.5, and a second error that was mine

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55
**Status:** open — the fix is not landed; listed in [OPEN.md](../OPEN.md)

## Context

`ledger.py:127` computes the width that travels with `--time-to-score`'s answer, and
`cli.py:1291`'s `_se_note` warns when that width is at or above P1's +5 pt target. The pair
exists because a crossing step read off a small subset is chosen by which rows fell where; it is
the mechanism that condemned `--eval-curve-n 20`'s default, and the number it printed there
(11.18 pt) is quoted in
[a default that cannot resolve its own effect](2026-09-08-a-default-that-cannot-resolve-its-own-effect.md)
and asserted in `tests/test_ledger.py:314`.

Found while computing the first real curve point's confidence interval by hand.

## The defect

```python
se = 100.0 * (0.25 / n) ** 0.5 if n else None
```

**`0.25` is `p(1-p)` at p=0.5, the worst case, and `p` is right there.** Every point carries
`correct` and `total`. The measured base is 0.874 and step 25 is 0.932, where the true SE at
n=500 is 1.09-1.13 pt against the printed 2.24 — the hardcode **inflates by 2.04x**, in the
conservative direction, at every rate other than 0.5.

## A second error I claimed and then withdrew

My first draft of this entry said the field was wrong twice: also that it is a single-point SE
where the question is a difference, needing a factor √2, and that the two errors partially
cancelled at 1.44x. **That half is wrong, and the mistake is mine rather than the code's.**

`time_to_score` compares each point's score against a **constant target** the caller passes.
One proportion against a constant is a single-point SE — no √2, no pairing. The quantity I
needed was a *different* question (two measured points compared, base vs step 25), and I read
the field as if it answered mine.

So the field has one error, not two, and it is conservative. What remains true and worth
recording is the shape of my error: **a width is defined by the comparison it is used for**, and
`se_pt` serves one caller (`_se_note`, point vs target) while I was asking another (point vs
point). Reading a width without checking which comparison produced it is how a correct field
becomes a wrong number in someone else's arithmetic — and I had already written the wrong
attribution into a draft entry before checking `time_to_score`'s own signature.

## Why the test could not see the real defect

`tests/test_ledger.py:314-318` asserts `se_pt == 11.18` at n=20 and `== 2.24` at n=500, **as
numbers**, deliberately, so a reader sees the width. The fixtures score 9/20 = **0.45** and
300/500 = **0.60**.

At those rates the true SE is 11.12 and 2.19, so the p=0.5 formula is off by **0.5% and 2.1%** —
inside any tolerance anyone would write, and matching to the two decimals the assertions pin.
**The fixture sits where the bug has no effect**: `p(1-p)` is flat near its maximum, so any test
data near 50% accuracy makes the hardcode nearly exact. The product it guards runs at 0.93.

Same shape as
[a dtype made the mechanism under test dead code](2026-09-08-a-dtype-made-the-mechanism-under-test-dead-code.md):
the fixture's value, not the assertion, decides whether the test can fail.

## What it actually costs

Not the `--eval-curve-n 20` verdict. At n=20 the printed 11.18 and the true 5.47 both exceed the
+5 pt target, so the default is condemned either way. That entry's conclusion stands; its
**number** is 2.04x too wide.

The cost is mid-range, where the warning's calibration decides:

| n | printed (p=0.5) | true at p=0.936 | warning fires | should fire |
|---:|---:|---:|---|---|
| 20 | 11.18 | 5.47 | yes | yes |
| 50 | 7.07 | 3.46 | **yes** | **no** |
| 100 | 5.00 | 2.45 | **yes** | **no** |
| 500 | 2.24 | 1.09 | no | no |

At n=50 and n=100 the warning fires on a subset that does resolve the effect, pushing a reader
toward rows they do not need — and a curve point costs **614.3 s, 114% of the training it
measures**, so an unnecessary widening is expensive.

## Fix

Not landed. `p = correct / total`, `se = 100 √(p(1-p)/n)`, from the crossing point rather than a
constant.

**Separately** — and this is an addition, not part of the fix — the saturation question does
need a difference width, and since #323 puts `eval-curve-<step>.jsonl` on disk the **paired**
McNemar form is computable: measured **1.43 pt** at 10.2% discordance against the unpaired
1.86 pt at the same two rates, so pairing is worth **1.30x**. That belongs to whatever reads
adjacent points, not to `time_to_score`, which does not compare two points at all.

## Rule

**A width is defined by the comparison it is used for.** Before quoting an SE, check whether it
was computed for a point against a constant, a point against another point, or a paired
difference — the three differ by up to √2 and by whether the pairing is available. I quoted one
into arithmetic that needed another, then wrote the mismatch up as a defect in the code.

And: **a test fixture near p=0.5 cannot detect a p=0.5 assumption.** For any formula with a
stationary point, pick fixture data away from it — here the product's own rate of ~0.9, where the
error is 2.04x rather than 0.5%.

