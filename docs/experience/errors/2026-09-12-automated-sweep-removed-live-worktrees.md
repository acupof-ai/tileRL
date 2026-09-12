# An automated disk sweep removed two owner-live worktrees: "clean and on origin" is not "not in use" — 2026-09-12

> Status: **fixed in the sweep procedure (report-only); no code shipped.** Two
> live worktrees were removed by a validation run of the daily disk-sweep job
> and restored within the minute with no data loss. The job was reworked to
> propose candidates and never delete. Reviewer/owner of the affected trees:
> 65 (`mmlu`) and 58 (`rb537d`).

## Context

The tileRL checkout had 54 registered git worktrees on the Mac (cap is
executors × 2 = 8). A disk-pressure order set up an automated daily sweep. The
first implementation classified a worktree for removal when it was

1. git-clean apart from a sync marker, and
2. its `HEAD` was contained in an `origin/*` ref (i.e. recoverable on GitHub).

A manual validation run, executed to confirm the scheduled job worked, removed
23 worktrees. Two of them were owner-declared **live**:

- `/private/tmp/claude-501/mmlu` (65, `bench/mmlu-thinking-spec`, PR #535)
- `/private/tmp/claude-501/rb537d` (58, `feat/spec-prefix-warm`, warm-path WIP)

Both passed the two predicates — clean trees, commits present on a remote or on
a local branch — so the script deleted them.

## Root cause

The predicate conflated *recoverable* with *not in use*. "The commit is on
origin" answers "can the bytes be retrieved?"; it says nothing about whether an
owner is actively working in that checkout right now. A live worktree is the
normal case for a small team sharing one checkout: it is clean, its branch is
pushed, and the owner is mid-task and intends to commit there next. A static
allowlist does not fix this — the *next* live tree is, by definition, not on
yesterday's allowlist — and a dynamic "is it on origin" check answers the wrong
question.

A second contributor: the dangerous behaviour was exercised by a one-shot
**validation run**, not by the schedule, so the keep-list that would have been
written for the live trees was still being drafted. The destructive path and
the "does it parse / fire" path were the same script.

## Fix

1. **The scheduled job never deletes.** `daily_sweep.sh` lists every worktree
   with size, branch and a KEEP/CANDIDATE reason (live pid, dirty,
   head-not-on-origin, or clean+on-origin) to a timestamped
   `candidates-<ts>.txt`, prunes only `__pycache__`/`.pytest_cache` and
   `uv cache prune`, and stops. A human removes a candidate by hand after the
   tree owner acks (or a3 rules) — the same gate the one-off sweep used.
2. An explicit `sweep.keep` lists current owner-live trees as defence in depth,
   but it is not the safety mechanism — report-only is.
3. The destructive one-off and the "does the schedule work" check are separate
   actions; a schedule is validated by reading its proposal, not by running its
   delete path.

Recovery in this incident was lossless: `git worktree remove` keeps the branch
ref, so `mmlu` (commits on origin) and `rb537d` (warm WIP already committed as
59cc03ec on a local branch) were restored with two `git worktree add` calls;
the registered count returned to 8. Nothing uncommitted was in either tree.

## Rule

Do not let an automated job infer liveness from git state. "Clean" and
"committed/pushed" are properties of the files; "in use" is a property of a
person, which the repository cannot report. Automate the inventory and the
measurement; gate the deletion on a human who asked the owner. A reaper that
deletes on a recoverability predicate will eventually delete the clean,
fully-pushed checkout somebody is working in.
