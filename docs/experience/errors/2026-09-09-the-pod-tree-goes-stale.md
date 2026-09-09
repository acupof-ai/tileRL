# The pod tree goes stale and still looks authoritative — 2026-09-09

## Context

A session's pod tree (`/work/tilerl-s-<session>`, synced by
`scripts/pod_sync.sh`) persists between sessions. Nobody re-syncs it on
entry, so a tree that was current when the session left it is stale by
the time anyone — including the same session, days later — goes back to
the pod and runs from it. The tree has no "last synced" marker, and the
code it contains is indistinguishable from a fresh checkout at a glance.

On 2026-09-09 a CI-verification attempt ran `pytest` on the pod tree at
`/work/tilerl-s-wt-main`. The tree was at an old revision: 409 tests
collected vs 541 on main, `tilerl` not installed in any python env on
the pod, 891 AppleDouble `._*` files from a macOS transfer. The attempt
produced no signal and had to fall back to local verification.

## Root cause

The tree is named by the session, not by the revision. `pod_sync.sh`
writes a tarball into `REMOTE_DIR` and wipes the old tree first, but
nothing re-runs the sync when a new session (or the same session, later)
attaches to the pod. The tree's mtime and contents look like a real
checkout; only the sha says otherwise, and nothing prints the sha
before running.

## Fix

`pod_run.sh` already prints `pod_run: tree <dir> sha <stamp>` before
anything that can fail (scripts/pod_run.sh:94), reading
`.synced_commit` written by `pod_sync.sh`. The gap is direct
invocation: anyone who `crictl exec`s into the pod and runs `pytest` or
`tilerl` by hand bypasses `pod_run.sh` and gets no sha. Two options:

1. A guard that reads `.synced_commit` on direct invocation — e.g. a
   `conftest.py` or a `tilerl` startup check that warns when the tree's
   sha is behind `origin/main`.
2. A shorter rule: never run from a pod tree by hand; always go through
   `pod_run.sh` or re-sync first.

Option 2 is the lazy fix and works if the convention holds; option 1
makes the convention structural.

## Rule

A long-lived checkout on a shared machine must carry its revision where
the next person to use it can see it before running. A tree that looks
current but is not is worse than no tree — it produces wrong numbers
with confidence.
