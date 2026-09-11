# A stale-branch squash reverted an already-merged PR (GDN kernel 4), docs and code

2026-09-11 · found by 65's #487 review, confirmed by a3 and 5f

## Context

#467 (GDN context-parallel conv-halo adjoint, kernel 4) merged at 54b0c895:
`cp_halo_bwd`, the window-aware `_gdn_backward_window`/prep reverse in
`tilerl_kernels/reference.py`, three lines in `backend.py` routing the op, the
world2 tape gate `tests/gdn_cp_halo_tape_world2.py`, and two wins entries.

#466 (checkpoint device faces) was cut from a base **before** #467 landed. Its
squash commit fadf4bd6 re-applied its own older tree over files #467 had since
changed. The overlapping paths — `reference.py`, `backend.py`, the halo tape
gate — resolved as a clean deletion of #467's hunks: `-186/+43` in
reference.py, the three backend lines gone, the gate file removed. It also
deleted `wins/2026-09-11-gdn-cp-halo-adjoint.md` and rewrote the tape entry
back to "kernel 4 pending".

## Root cause

A squash built against a stale base carries the **whole file content** of its
branch. When that content predates a merge on the same path, applying the
squash overwrites the newer file wholesale — and git reports no conflict,
because the squash author's resolution is already baked into the blob. CI ran
green: the deleted gate was not in its test set (it is a `tests/*_world2.py`
script, not collected by pytest), the deleted wins entry's dangling CHANGELOG
link was not yet covered at that revision, and the removed op had no other
caller that fails to import — the tape simply stopped routing the halo
gradient. The defect was invisible to every automated check present.

The stale check that existed looked only at whether the PR *branch* was behind
main; being mergeable (no textual conflict) was treated as current, which is
exactly the case a whole-file squash defeats.

## Fix

Restore from the known-good merge 54b0c895 in one PR (#487):
`git checkout 54b0c895 -- reference.py backend.py tests/gdn_cp_halo_tape_world2.py`
plus both wins entries. Verified per file that 54b0c895..origin/main touched
each path only via fadf4bd6 (the bad squash), so the restore drops no later
work — #466's real changes live in `model.py`/`precision.py`/`cli.py`, not in
the reverted hunks. Gate green locally: `gdn_cp_halo_tape_world2.py`
(worst 2.26e-3 vs the recorded ≤2.3e-3) and `gdn_halo_world2.py` (all-zero
rel), and re-run on cards 4/5 before merge.

## Rule

Before merging ANY PR, run the per-file stale check against what is actually on
main, not just the mergeable flag:

```bash
git log --oneline <merge-base>..origin/main -- <each file the PR touches>
```

A file with a commit the branch does not contain must be reviewed hunk-by-hunk
in the squash/rebase — a whole-file resolution that predates it silently
reverts that commit. Whole-file squashes over shared paths are the dangerous
case; prefer rebase-with-rerere or a merge that preserves intervening blobs.
