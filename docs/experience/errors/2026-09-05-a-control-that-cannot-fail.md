# A control that cannot fail — eight instances in one day, 2026-09-05

**Date:** 2026-09-05
**Sessions:** tilerl-25, tilerl-48 and v100-sm70-fp4-55, independently
**Task:** negative controls on five unrelated changes

## Context

A negative control is supposed to answer "would this test notice if the thing it
guards were broken?". Eight times in one day, across three sessions, the control
itself was broken in a way that made it answer yes when the honest answer was no.
None was caught by reading. Four were caught by deleting the guard and watching
the test pass anyway; the fifth by a loader refusing the file, the sixth by running
the same command under a different shell, the seventh by replacing the tensor
the control looked at, and the eighth by a reviewer reading the code path and
saying the counter I quoted cannot move.

**The eighth is the inverse of the other seven and is listed with them because the
lesson generalizes, not because the mechanism matches.** Seven controls passed when
they should have failed. The eighth *failed* — correctly, on a real defect — but by a
different assertion than the one carrying the claim, so the red looked like proof of
something it never tested. That is harder to catch, because every habit built for the
first seven treats red as the end of the check.

## The instances

**1. A second module copy (`importlib`).** `_timing_snapshot` loads
`bench_harness.py` through `importlib.util.spec_from_file_location`, which builds
a *new* module object. The test monkeypatched `bh._BASELINE` on the imported
`bench_harness` — a different object than the one the function executes. The
patch reached nothing, the test asserted against a file the code never touched,
and it passed with the guard removed. Fixed by asserting on the real `_BASELINE`.

**2. `.pyc` cache.** A file restored on disk is not a module restored in memory:
the control ran against the cached bytecode of the version it thought it had
removed.

**3. A reaping assertion in `pod_run.sh`** that held whether or not the parent
waited.

**4. Warming a store to a block-aligned length.** v100's prefix test warmed the
store to 144 tokens. Because 144 is block-aligned, the whole-prompt publish path
fired too — so swapping the publish gate from `% 16` to `% 64` still left an
entry to hit and the test passed both ways. Rewarming to **150** isolated the
mid-chunk path: `% 16` publishes 3, `% 64` publishes **0**, and the test fails on
`assert 0 >= 1`.

**5. A re-measurement that inherited the original's categories.** Two sessions
bucketed the same `adapter.safetensors` into live/dead and agreed to the last
digit. The second script hardcoded `DEAD = (".scale", ".conv1d")` — the two
suffixes the first had reported — and took "live" as the complement. That is not
an independent measurement of the partition, it is a re-execution of the same
partition on the same bytes, and it could only agree. The file also contains
**466 `.wscale` adapters**, in neither bucket. Grouping by *every* suffix present,
rather than by the categories already believed in, showed it in one run: live is
996 keys, not 1462. What actually caught it was `--load-adapter` refusing with
"1090 unknown keys" — a gate, not either measurement.

**6. A shell that swallowed the exit code.** A CI step ran a gate in a subshell
under zsh; `set -e` there ate the gate's exit 1 and the step reported 0. The gate
detected the defect, printed it, and the harness above it called that a pass. Run
the check under the shell CI uses (bash), and assert on `$?` of the process
itself — a grep or a pipe in between reports its own status, not the gate's.

**7. A compared tensor with a local term.** The guard: an ungrouped TP all-reduce
destroys DP, so dp0's and dp1's gradients must differ. They did differ — max|d|
3.4e+02 — with the bug live. `embed_tokens`' gradient carries a scatter over each
replica's own ids, so it differs between replicas whether or not the collective is
grouped correctly. The control passed on a local term that had nothing to do with
the defect. Fixed by asserting the collective itself on a rank-valued probe, which
has no local term to hide behind: grouped gives `{0:1, 1:1, 2:5, 3:5}`, ungrouped
gives `{0:6, 1:6, 2:6, 3:6}`, and the control then fails.

This is the only one of the seven caught **before** shipping, and only because the
control was run at all. The gate itself was green.

## Root cause

They share a shape. In each, the control's *setup* silently satisfied the
condition under test by a second route:

| | the guard | the route that kept passing |
|---|---|---|
| 1 | patched `_BASELINE` | a second module object |
| 2 | restored source file | cached bytecode |
| 3 | parent waits | assertion true either way |
| 4 | mid-chunk publish | whole-prompt publish |
| 5 | re-measure independently | the re-measurement inherited the categories |
| 6 | the gate's exit code | the shell wrapping it returned its own |
| 7 | dp0 vs dp1 gradients differ | a local scatter term differs anyway |

Reading the test cannot find this, because the test reads correctly. Only running
it in the state it claims to detect can.

**Every one of the seven was downstream of the defect** — the assertion sat
somewhere the bug's effect could be masked by something else contributing to the
same number. Distance from the defect is the risk factor. A control has to be
sensitive to the defect, not merely downstream of it: in instance 7 the shortest
distance was one line, reducing a tensor whose correct value is known per rank.

### Instance 8: the control failed, and the assertion that failed was not the one under test

| | the claim | what actually went red |
|---|---|---|
| 8 | "deleting the rollback re-indexes the dropped prefix (`ssd_entries` 0 → 1)" | `assert not left` — a file on disk, which **both** candidate mutations leave |

`_flush_loop` catches a `drop()` that lands mid-save with two separate lines: a
`still_pending = table.get(k) is blob` check that keeps the key out of the index, and an
`os.remove(dst)` that removes the bytes. I mutated the whole `if still_pending: … else:
remove` block, saw the resurrection, and published it as the cost of the `os.remove` line.
Measured apart:

| mutation | `ssd_entries` | files left |
|---|---:|---:|
| `still_pending` forced True | **1** | 1 |
| `os.remove(dst)` deleted only | **0** | 1 |

So the claim belonged to the other guard. Two things had to go wrong together, and both are
generic:

* **the mutation was wider than the claim** — one edit removed two guards, and I attributed
  the damage to the line my commit message happened to name;
* **assertion order decided what was testable** — both mutations leave a file, the file
  assertion came first, and pytest stops at the first failure, so the assertion whose message
  said "indexed again" **never executed** in the run I cited as its proof.

Caught on PR review by codex reading the code path: `still_pending` is already false after a
`drop()`, so `_track_written` is skipped and the counter cannot move. Confirmed independently
by tilerl-25 re-running all four mutations of that PR.

## Fix

Run every negative control **in the failing state at least once**, and record that
it failed. A control that has never been observed to fail is evidence about the
control, not about the code.

Concretely: delete the guard, clear `__pycache__`, run the
test, confirm FAIL, restore, confirm PASS. In this session that turned up two
vacuous controls out of five written.

**And read which assertion failed.** Instance 8 is here because the paragraph above was
followed exactly and was not enough: the control ran in the failing state, it failed, and it
failed by an assertion that fires in every arm. So the check has three parts, not one — mutate
one line per claim; order the assertions so the discriminating one comes first and anything that
fails in every arm comes after it; then read the assertion text in the failure output and
confirm it is the one you are about to cite. A red test names *an* assertion, not *your*
assertion.

And run it under the shell that will run it in CI. The sixth instance never touched
the test at all: the assertion was correct, the failure was printed, and the step
still reported success because the shell between them returned its own status.

For a *measurement* rather than a test, the same rule reads: enumerate what is
actually there instead of counting the categories you already have. Agreement
between two derivations that share a premise tests the arithmetic, not the premise.

## Rule

**A negative control that has not been seen to fail is not a control.** Watching
a test pass proves nothing about what it would catch; the only evidence is a run
where it caught something. Budget the extra run — it is one command, and both
vacuous controls here would have shipped as proof otherwise.

**And a control seen to fail is not yet evidence for the claim it was written for.**
That is instance 8: the failure was real and named a different assertion. Red is where
the check starts, not where it ends.

Related, from the same day:
[the gate cannot see the number it guards](2026-09-05-the-gate-cannot-see-the-number-it-guards.md)
is the adjacent failure — a control that runs correctly but at a tolerance too
coarse to resolve what it guards.
