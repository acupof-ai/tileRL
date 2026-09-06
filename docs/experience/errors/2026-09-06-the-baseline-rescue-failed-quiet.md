# The baseline rescue failed quiet, and quiet meant losing the row — 2026-09-06

> Status: Shipped. `pull()` returns nonzero, the sync line has no `|| true`, and
> `tests/test_baseline_rescue.py` gates both halves plus the loss itself.

## Context

`pod_sync.sh` exists to put this checkout on the pod. It does three things in order:

1. `baseline.py pull` — merge the pod's `bench-baseline.json` rows into this tree
2. wipe the remote checkout
3. extract this tree's tarball over it

Step 1 is a rescue: the pod raises bench rows, and step 3 would otherwise overwrite them
with a local file that never saw them. I was reading the script for the `runs/` exemption
([the OOM was micro=0](2026-09-06-the-oom-was-micro-zero.md)) and the rescue's own line
looked wrong.

## Root Cause

**Two halves, and the second made the first invisible.**

`baseline.pull()` ran `subprocess.run([str(Path.home() / "bin/pod"), ...])` with no guard.
On this Mac `~/bin/pod` is a **symlink into another repository** (`code/aupai/scripts/pod`),
so it is absent whenever that repo moves — and `subprocess.run` on a missing executable
**raises `FileNotFoundError`** rather than returning non-zero. Measured, not assumed: with
`Path.home()` pointed at an empty directory, the unguarded `pull()` exits with an uncaught
`FileNotFoundError`.

The caller was `... baseline.py pull >/dev/null 2>&1 || true`. That swallows a raise and a
returned 1 identically, and `2>&1` discards the traceback that would have named the cause.
So the rescue could fail on every invocation and print nothing.

**The wipe is not what loses the row — step 3 is.** `bench-baseline.json` is tracked, so it
travels in the tarball; the wipe deletes the pod's copy and the extract replaces it with
the local one. Simulated all three steps with a row the local tree has never seen:

| | pod ends with |
|---|---|
| pull works | **2 rows**, `d/a` at **140.0** |
| pull fails silently | **1 row**, `d/a` at **100.0** |

The pod's 140.0 and its second row are gone, and the exit status is 0.

## Fix

`pull()` catches `OSError` and returns 1 with the launcher path on stderr. The sync line
drops `|| true` and its `2>&1`, so `set -euo pipefail` aborts the sync **before** the wipe.
`SKIP_BASELINE_PULL=1` still short-circuits the `[ ... ] ||` test, so the documented
deliberate overwrite is unchanged.

Checked that `set -e` actually fires here, because an `||` list's failure only trips it
when the failing command is last: with the rhs exiting 3 the script stops at rc=3 and
never reaches the next line; with `SKIP_BASELINE_PULL=1` it reaches it; with the old
`|| true` appended it reaches it. Three arms, and the third is the defect reproduced.

## The gates, and the routes they pass by

`tests/test_baseline_rescue.py`, 6 tests, two red controls:

- `pull()` returns `1` with the launcher missing — control: swapping the `except OSError`
  line makes it `raised:FileNotFoundError`. The control asserts the **exception's name**,
  not just a non-zero exit, because "it exited non-zero" is also true of an import error
  and would pass by the wrong route.
- the sync line does not reach the wipe when the pull fails — control: the same line with
  `|| true` restored does reach it. Verified the abort's route: `python3` **is** on the
  PATH the test builds and the fake `baseline.py` **does** exit 1, so the missing
  "REACHED" is the pull's failure and not a missing interpreter.
- `SKIP_BASELINE_PULL=1` still reaches the wipe.
- the loss itself, asserted on the artifact: 140.0/2 rows against 100.0/1 row.

## Rule

**`subprocess.run` on a missing executable raises; it does not return non-zero.** A
caller that only checks `returncode` has no path for the launcher being absent, and
`|| true` in the shell above it turns that into silence.

Second, and the reason this was worth finding: **a rescue that fails quiet is worse than
no rescue**, because the thing it protects is exactly what the next step overwrites. When
a step exists to preserve data, its failure has to stop the pipeline, not be tolerated by
it — the `|| true` was there to keep a cosmetic failure from blocking a sync, and it kept
a data-loss failure from blocking one too.

## Results

No perf change. `bench-baseline.json` integrity only; no measurement to report.
