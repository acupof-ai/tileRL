# Curve points record their own churn — 2026-09-09

**Status:** pending-remote. The churn function is unit-checked in `python -m tilerl.ledger`
(cell G, mutation verified red); the field populating in a real run's manifest needs a
run, which this machine cannot start.

## Context

Twice today we needed "how many questions flipped between two curve points" and twice
someone wrote a script after the fact to recover it from `eval-curve-*.jsonl`: cc's
`scripts/curve_table.py` for the eight-cell per-question table, and 9b's pricing script.

The expensive instance was the false alarm: `b78ae28`'s headline "37 questions changed
hands between step 50 and 100" was nearly dismissed as noise, because nobody knew the
instrument's own flip rate. The answer was in the data all along — adjacent points on the
plateau flip 15/15/16 questions — but nobody had computed it, and a claim about noise
without the noise floor next to it reads as either alarming or dismissable depending on
the reader's mood.

## What changed

Every curve point now records, against the **previous** point in the same run:

- `churn`: total flips in both directions;
- `churn_dir`: `[right→wrong, wrong→right]` (the net is the score difference, already
  recorded, so it is not repeated).

Zero new evals: the per-problem rows are written for every point since #323, and the
previous point's rows sit in the same run directory. Pairing is by row position, valid
within one run (`curve_rows` is sliced once outside the loop, `per_problem` is written in
input order); it is invalid across runs, and the docstring says so. The first point has
no predecessor and records `null`, not 0 — 0 means "no flips", null means "no comparable
point". Different row counts (a changed `--eval-curve-n`, a truncated file) also record
null and log why.

## Rule

A run reports its own noise floor for free. The next "these two points differ by N
questions" claim has N's instrument beside it in the same manifest — no script, no
hour-long control run, no argument about whether N is big.

## Evidence

`python -m tilerl.ledger` cell G: two row sets with one flip each direction asserts
`(1, 1)`; no predecessor and mismatched lengths assert null. Mutation control
(`return 0, 0`) verified red and reverted green. `ruff check` clean; CPU suite
517 passed / 15 skipped / 6 xfailed.
