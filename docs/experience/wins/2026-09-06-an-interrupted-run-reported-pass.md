# An interrupted run reported `pass` — cpu, 2026-09-06

> Status: fixed here. Closes the bound documented in
> [per-rollout logging](2026-09-06-per-rollout-length-reward.md), which recorded
> the misreading and deliberately left it.

## Context

Making an interrupted run visible to the ledger created a way to misread it.
Before the pre-loop `write_manifest`, a killed run left no manifest and
`tilerl ledger` returned `[]`. After it, the run appears — and prints a verdict
it never earned:

```
e069c8ff28b7  train  running    pass
```

The OPD path is the one that reaches `pass`: grpo pre-seeds the
`rollouts_within_cap` gate, so its gate list is non-empty and `format_run` reads
`skip`, but OPD appends nothing, and `gates_pass([])` is `all([])` = `True`.

## What Worked

`format_run` checks `finished` before any gate. Gates are written by `_finish`,
so an unfinished manifest carries only what was pre-seeded, and no gate has been
evaluated — the verdict column now says **`killed`** rather than a judgement.

Measured on cpu, real `tilerl train --opd`, SIGTERM by a pid verified through
`ps -o command=`, same run as before the change:

| | verdict |
|---|---|
| before | `pass` |
| after | `killed` |

Display layer only. `gates_pass` is unchanged and `_finish`'s exit code with it,
because `_finish` sets `finished` before it formats — a finished run's verdict is
computed exactly as it was.

**The existing gate asserted the defect.** `test_manifest_round_trip_and_lineage`
built a manifest, never set `finished`, and asserted `format_run(...) == "pass"`.
That line passed throughout and encoded the bug as the expectation, so the fix had
to change it: the unfinished arm now asserts `killed`, and a second arm sets
`finished` and asserts `pass`. Negative control — dropping the `finished` check —
fails on that first arm with `assert 'pass' == 'killed'`.

## Rule

A verdict computed from an empty collection of checks is a verdict about nothing;
`all([])` is `True`, so the absence of gates reads as consent. When a status field
and a judgement field can disagree, make the status gate the judgement rather than
sit beside it — and check whether an existing test asserts the current wrong
answer before assuming it protects you.

## Results

`341 passed, 14 skipped` on cpu, ruff clean. No perf surface: `format_run` is one
print per row in `tilerl ledger`.
