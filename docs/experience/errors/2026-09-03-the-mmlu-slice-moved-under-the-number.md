---
question: MMLU 0-shot reads 74.6% today against 76.3% on record. Regression or harness?
status: measured
source: H20 sm90, 27B NVFP4, `scripts/mmlu.py` unchanged, clean main
---

# The recorded 76.3% and today's 74.6% score different questions

`docs/experience/wins/2026-08-28-mmlu-letter-restricted.md` records MMLU 0-shot
**763/1000 = 76.3%**. Its own runner, `scripts/mmlu.py --engine tilerl --n
1000`, run today on a clean checkout, returns **746/1000 = 74.6%**.

It is not a regression. **The two runs scored different questions** — the
sampled index sets share **2 of 1000**.

## Root cause

The eval slice is drawn at run time from the dataset length, and the expression
that draws it changed.

```python
# e4d6565, 2026-08-28 — the draw behind the recorded 76.3%
idx = list(range(len(ds)))
random.Random(seed).shuffle(idx)
idx = sorted(idx[:n])

# 81e9789, 2026-08-29 — one line instead, and a different 1000 questions
idx = sorted(random.Random(seed).sample(range(len(ds)), n))
```

`shuffle`-then-take and `sample` are different draws from the same seed. Both
are deterministic, both are "the fixed 0-shot MMLU slice" in the docstring, and
nothing in the diff or the review said the eval had been re-rolled.
`81e9789` is *feat(bench): accuracy suite* — the commit that made MMLU a gate is
the commit that moved what MMLU measures.

Verified against the 08-28 run's own saved artifact (`/work/mmlu_tilerl.json`,
which stores its `idx`):

| check | result |
|---|---|
| `shuffle`-then-take, seed 0, reproduces the recorded 08-28 index list | **True** |
| current `sample(range(N), 1000)`, seed 0, equals it | False, overlap **2/1000** |
| today's dataset re-read at the 08-28 indices reproduces the 08-28 gold letters | **True** |

The last row is what rules out the dataset: the rows are unchanged, `len(ds)` is
14042 in both, only the draw moved. No `len(ds)` in [13000, 60000) reproduces
the old indices under `sample`, which is what sent us to the diff.

## What the numbers are

| harness | engine config | slice | score |
|---|---|---|---:|
| `scripts/mmlu.py`, 2026-08-28 | fused projections, `PrefixStore`, decode graph, concurrency 32 | shuffle-draw | 763/1000 = **76.3%** |
| `scripts/mmlu.py`, today, clean main | same | sample-draw | 746/1000 = **74.6%** |
| `eval.mmlu_accuracy`, today (training path) | unfused, `NoPrefixStore`, eager, concurrency 8 | sample-draw | 742/1000 = **74.2%** |

Two independent 1000-question samples of MMLU have a standard error near 1.4
points each, so 76.3 and 74.6 are one sample apart and say nothing about the
model. The engine-config difference, measured on one slice, is 4 questions.

`README.md` and `docs/roadmap.md` carried 76.3% as the project's MMLU number.
Today's tree cannot reproduce it — not because the model got worse but because
the question set no longer exists in the code. Both now carry 74.6% with the
draw named.

## Rule

An eval slice sampled from `len(dataset)` at run time is not fixed; it is
fixed only against one expression, one seed and one dataset length, and a
refactor of any of the three silently re-rolls it. Two accuracy numbers are
comparable only if the same questions were asked.

Record the slice with the score. The 08-28 run saved its `idx` and that artifact
is the only reason this took twenty minutes instead of a bisect — every eval
that writes a number should write the index list or a hash of it beside it.

A drop of 1-2 points on a 1000-question suite is inside sampling noise between
two draws. Check that both arms asked the same questions before spending a
bisect on the model.
