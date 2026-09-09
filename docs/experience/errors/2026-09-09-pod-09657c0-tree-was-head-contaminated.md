# The pod's 09657c0 tree was byte-identical to HEAD — every baseline number void

**Date:** 2026-09-09
**Arch:** H20 (sm90) card 6

## Context

The 135.5-vs-126.5 throughput gap investigation compared a "09657c0 baseline"
against HEAD across a day of profiling: tick timing, a five-bucket prefill
profile, the fp8 dispatch audit, the spec-arm wall A/B. Every 09657c0 number
came from `/work/tilerl-s-wt-09657c0` on the pod.

That tree was not 09657c0. `diff -rq` against the HEAD tree
(`/work/tilerl-s-wt-spec-h20`) found the source **byte-identical** — only
`__pycache__` differed. Two independent markers confirm it:

| marker | true 09657c0 | pod "09657c0" tree |
|---|---|---|
| `draft.attach` signature | `draft.attach(backend, kv_pool.num_blocks)` (engine.py:347) | `draft.attach(backend, kv_pool.num_blocks, dtype=kv_pool.dtype)` (engine.py:462, the HEAD form) |
| `_resolve` refs in backend.py | 15 | 19 |
| `_draft_step_timed` in engine.py | absent | present |

The local checkout `/private/tmp/claude-501/wt-09657c0` (96aed09e = 09657c0 +
3 bench scripts, src/ untouched) is the true baseline; the pod copy had been
overwritten by a HEAD-based sync at some point.

## Root Cause

The pod trees are tarballs, not clones — they have no `.git`, so a tree's
revision is invisible unless something stamps it. `pod_sync` already stamps
`.synced_commit` at the repo root, and `benchrec.git_commit()` reads it. **The
profile scripts bypassed benchrec entirely** — they are one-off scripts that
print a table and exit, so nothing read the stamp, and the contamination was
invisible for a full day.

The deeper lesson is not "add a second revision file." A second parallel
mechanism ends up bypassed exactly the way `.synced_commit` was: the profile
scripts ran outside the store that carries the provenance. The fix is to make
every measurement script call `benchrec.git_commit()` / `git_dirty()` and
print the sha in its header — ten lines — so a run's tree is checkable at a
glance. The re-run of the spec profile does this.

## Fix

1. Re-synced the true 09657c0 baseline (96aed09e) to the pod; verified
   `draft.attach`, `_resolve` count, and `.synced_commit` all match the real
   commit.
2. `scripts/acc_spec_prefill_profile.py` now imports `benchrec` and prints
   `git_commit()` / `git_dirty()` in the table header and the JSON.
3. The 09657c0 columns in the tick-timing entry are marked VOID, not deleted —
   deleting erases the evidence; marking leaves a checkable lesson.

## What is void and what survives

**Void** (measured on a tree byte-identical to HEAD, so every "09657c0 vs HEAD"
difference was run-to-run noise between identical-code runs):

- The W=8 tick "improvement" 43.52 → 41.96 ms (3.6%).
- The prefill kernel "flat" 290.9 → 288.0 ms/q and per-chunk 123.8 → 118.4 ms.
- The base-arm wall "flat" 211.0 → 209.9 s.
- The spec-arm "4% regression" 131.7 → 126.5 tok/s (200q) and 123.1 → 128.4 s (50q).
- The fp8→bf16 hypothesis test: both trees were HEAD, so "all-fp8 at both
  shas" was trivially true and proved nothing about 09657c0.
- The two-instrument disagreement at "09657c0" (7.2-9.1 vs 14.55 s): it was
  HEAD vs HEAD, and the "outlier" reading was noise, not a timing-boundary
  phenomenon.

**Survives** (does not depend on the 09657c0 tree):

- 135.5 tok/s has never run on a main sha — the recorded sha (40bc83c) is
  B=8-hardwired, and B=1 support landed in #58. This is a provenance fact
  about the recorded number, not a measurement.
- The arm-order artifact: whichever arm runs first pays a one-time prefill
  cost. This is a within-tree comparison and is real, but it is now a
  conclusion about HEAD, not about 09657c0.
- The 6x eager-decode fallback (same-sha A/B) and the #389 loud-fallback fix.
- The prefix-reuse hypothesis being dead by construction (the harness builds
  with `NoPrefixStore` at any sha).

## Rule

A measurement is only as good as the tree it ran on, and a tarball's tree is
invisible until something stamps it. The provenance mechanism (`benchrec`,
`.synced_commit`) already existed — the failure was running the day's most
important measurement *outside* it. **Every script that prints a performance
number must print the sha it ran on, by calling `benchrec.git_commit()` /
`git_dirty()` in its header.** A second parallel revision file is not the fix;
it would be bypassed the same way.
