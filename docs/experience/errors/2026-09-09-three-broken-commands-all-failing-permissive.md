# Three broken commands, all failing permissive, all read as evidence — 2026-09-09

## Context

A merge-completeness check was built to catch "silent reverts" — PRs whose
three-way merge drops changes main made since the branch's base. Three
incidents were cited as evidence that this happens: #364 ate #357's
`--deterministic`, #342 would eat #367's `pytest-xdist`, #384 would eat
#383's fix. All three were real merges with green CI.

The check was implemented (PR #390) and a peer verified `git merge-tree
--write-tree` across five change shapes: it either merges correctly (both
sides' changes present) or reports a conflict. It never silently drops one
side's additions. That finding was correct. The mechanism story was not.

## Root cause: three instrument failures, all returning empty

### 1. `git show <tree>:<path>` printed the root directory, not the file

```
$ git show "$tree:src/tilerl/cli.py" | head -3
tree 22f5395a55a47536f1c78c528373e311692598b9
.dockerignore
.gitattributes
```

The command printed the tree's root listing. `grep -c deterministic`
returned 0 — read as "the line is absent from the merge result." The
positive control that would have caught it: `grep -c "def main"` on the
same output — also 0, proving the output was not the file.

### 2. OID concatenation

A later read produced `fatal: Not a valid object name 733A...F3v.lock` —
the tree OID had a path fragment glued to it. Same 0, same "absent"
reading, same missing positive control.

### 3. `git merge-base --is-ancestor <sha> <squash-commit>^2`

A squash commit has one parent. `^2` does not exist, the command errors,
and the `||` branch printed "branch did NOT contain the commit." That
string was error handling, not a judgment.

### Why they corroborated

All three failed in the same direction — returning empty, 0, or an error
that was caught and read as "not present." Three independent instruments
agreeing felt like confirmation. They were three instances of the same
failure mode: a reading that returns "nothing" was trusted without a
positive control proving the instrument can return "something."

## The actual cause of #364

```
$ git merge-base --is-ancestor 5cd9404 d0bb05af   # #364 branch head
YES

$ git show d0bb05af -- src/tilerl/cli.py | grep '^-.*deterministic'
-                          decode_graph=not args.deterministic,
-    p_train.add_argument("--deterministic", action="store_true",
```

The #364 branch contained #357. The deletion was in #364's own diff,
visible to anyone who opened it. No stale base, no three-way merge artifact,
no GitHub squash behavior. The author deleted the flag; the reviewer saw
two green checks and merged.

## The one solid experiment

The peer's five-shape verification of `git merge-tree --write-tree` is the
only measurement that survived. It proved a negative — "merge-tree does not
silently drop lines" — across same-line edits, adjacent insertions, context
deletion, and hunk-boundary interaction. A negative is the claim that most
needs a positive control, and the five shapes were that control: each shape
was constructed to produce a drop if one were possible.

## The broader pattern: field semantics that are not what they appear

Three times today, a field's apparent meaning was not its actual behavior:

1. `git status: clean` — means no uncommitted changes, not "up to date with
   the remote." A branch can be clean and hours behind.
2. `mergeStateStatus: CLEAN` — means no merge conflicts on GitHub's merge
   preview, not "CI passed." A PR with zero checks reports CLEAN.
3. `cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}` — the
   expression likely stringifies to `"false"` (truthy) in the concurrency
   block, so main runs were still cancelled.

The shared fix shape: do not debug how the ambiguous field is interpreted —
replace it with a mechanism that has no ambiguity. For the concurrency case,
make main's group unique per SHA (`group: ci-${{ github.ref }}${{ github.sha }}`)
so there is nothing to cancel, rather than arguing with boolean coercion.
`group` is a string field; expression interpolation into a string is its
normal semantics.

## Rule

A reading that returns "0 / empty / doesn't exist" must pass a positive
control — a probe that must return non-zero — before it is used as evidence.
Three permissive instruments agreeing is not confirmation; it is the same
blind spot three times.
