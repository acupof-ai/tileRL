# A finished run with no gates read `pass` — cpu, 2026-09-06

> Status: fixed here. Closes the follow-up flagged in
> [an interrupted run reported pass](2026-09-06-an-interrupted-run-reported-pass.md),
> which fixed the unfinished case and left this one open pending a naming decision.

## Context

`gates_pass([])` is `all([])` = `True`. #148 made an unfinished run print `killed`
instead of `pass`, but a *finished* manifest with an empty gate list still fell
through to the pass/FAIL branch and printed `pass` — a judgement over zero checks.

The producer was found by probe rather than by reading: a real `tilerl merge` on
cpu with two 4×4 safetensors and a `config.json`,

```
144b31c31f4d  merge  2026-09-05T21:44:19+00:00 pass   tensors=1
manifest: command=merge  gates=[]  finished=2026-09-05T21:44:19+00:00
```

`cmd_merge` builds its manifest, sets `finished`, and calls `write_manifest`
directly — never `_finish` — so it appends no gates. Every `tilerl merge` row on
disk read `pass`.

`grep -n gates src/tilerl/{cli,ledger}.py` confirms it is the only producer: gates
are written at `cli.py:594` (grpo pre-seeds drift), `cli.py:806` (`_finish`, inside
the `finished` guard), and `ledger.py:44` (the empty init).

## What Worked

A fourth verdict, `none`, for finished-with-no-gates-defined. Not `skip`: in this
tree `skip` means a gate existed and was suppressed — `gates_skip_after`, and the
drift gate that `tilerl train --allow-short-rollouts` bypasses — and merge defines
none, so the two states stay distinguishable. The naming was ckl's call, not mine; I
had leaned `skip` and was overruled, correctly, for that reason.

`gates_pass` is untouched, so `_finish`'s exit code is untouched. Display only.

Four negative controls, each red on its own assertion:

| broken | fired |
|---|---|
| restore main's two-branch shape | `pass   assert 'pass' == 'none'` |
| name it `skip` | `assert 'skip' == 'none'` |
| let the gateless branch swallow all-skipped | `assert 'pass' == 'skip'`, in two *other* tests |
| break `gates_pass` | `assert 1 == 0` (the exit code), outside merge |

Controls 3 and 4 fail different tests than 1 and 2, which is what says the new branch
does not absorb its neighbours and that `gates_pass` is still independently guarded.

**My first attempt at control 1 was invalid.** I replaced `elif not m["gates"]:` with
`elif False:`, which drops through to the branch I had just edited — where the
`m["gates"] and` guard was gone, so `all([])` is True and it printed `skip`. That is a
third state I had constructed, not main's behaviour. Restoring main's exact two
branches reproduces `pass`. A control that mutates one line of a rewritten block does
not restore the old behaviour; it makes a new one.

## Rule

When a control is supposed to reproduce the old behaviour, restore the old code, not a
mutation of the new code. Disabling one branch of an edited conditional leaves the
other branches in their edited form, so the "before" arm measures a state that never
existed — and it will still go red, which is what makes it convincing.

## Results

`344 passed, 14 skipped` on cpu, ruff clean. No perf surface: one branch in
`format_run`, one print per `tilerl ledger` row.

The end-to-end assertion lives in `test_merge.py`, on the existing test that already
runs `tilerl merge` through `main()` — the behaviour is asserted where it is produced,
not in a new fixture. `test_ledger.py` covers the cell directly and asserts
`gates_pass([])` is still True, so a future change to the exit code cannot hide behind
this one.
