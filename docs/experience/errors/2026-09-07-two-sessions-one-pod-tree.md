# Two sessions, one pod tree — 2026-09-07

## Context

`scripts/pod_sync.sh` wipes the remote checkout before extracting the tarball
(`find . -mindepth 1 ! -path './runs' ! -path './runs/*' -delete`), and every
session synced into the same `/work/tilerl`. Two sessions staging different
branches there therefore reverted each other, and the second one's probe then
measured the reverted tree while reporting its own sha. On this day one session
synced over another's setup; nothing was lost, by luck rather than by structure.

## Root Cause

The tree was named by the tool, not by the caller. `REMOTE_DIR` defaulted to a
single constant, so isolation depended on nobody else running at the same time —
a convention, and conventions do not hold across sessions that cannot see each
other. Two consequences, both silent: the wipe reached another session's staged
branch, and a number in a log had no sha attached to say which tree produced it.

## Fix

`REMOTE_DIR` defaults to `/work/tilerl-s-<session>`, the session name from the
worktree's basename (`scripts/pod_session.sh`). The wipe already runs inside
`REMOTE_DIR`, so confining the tree confines the wipe — the fix is the default,
not a narrower `find`. `pod_run.sh`'s runner prints
`pod_run: tree <dir> sha <stamp>` before anything that can fail.

`bench-baseline.json` moves to `/work/tilerl-baseline/`, outside every session
tree, and `bench_harness` **merges** into it: two sessions each hold a snapshot
taken before the other wrote, so a plain write drops the other's rows. That is
the same loss, moved into the one file the sessions still share.

Three details that were wrong in the first draft and are the reason this entry
exists rather than a one-line commit:

**`/work/tilerl-<session>` was not safe.** `/work` already held **29 ad-hoc
`tilerl-<word>` trees** from hand-set `REMOTE_DIR` overrides — `tilerl-fix`,
`tilerl-main`, `tilerl-spec`, `tilerl-train`, `tilerl-clean`, `tilerl-base` —
and several carried a live `.synced_commit` (fix `a702c9a`, spec `0fd8f7c`,
train `fa97b13`). A session named `fix` would have synced into one and wiped it:
the collision the change removes, reintroduced by the change. Hence `s-`.

**`basename | tr -c` appended a dash to every name.** `tr` translates the
trailing newline along with everything else, so the tree was `tilerl-s-<name>-`.
Caught by asserting the exact string rather than a prefix.

**The first default-tree assertion could not fail.** Every other arm passes
`REMOTE_DIR` explicitly, so reverting `pod_run.sh` to the shared `/work/tilerl`
left all five arms green. Only running that mutant showed it. The arm now reads
the default with `REMOTE_DIR` unset.

## Rule

A tool that destroys state must name its target from the caller, not from a
constant. And when a fix moves a shared resource, enumerate what already lives
at the new name — the 29 trees were readable in one `ls` before the default was
written, and reading them is what stopped the fix from reproducing the bug.
