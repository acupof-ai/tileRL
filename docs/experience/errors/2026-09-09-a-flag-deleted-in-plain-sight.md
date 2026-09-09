# A flag deleted in plain sight — `--deterministic` removed by a PR whose diff showed the deletion

## Context

`--deterministic` landed in #357 (5cd9404, 2026-09-09 20:01) to make rollout
A/B comparisons bitwise-reproducible: it sets `decode_graph=False`, turning
off the captured decode graph that is the proven cross-process nondeterminism
source (16/24 rollout rows differ with the graph on, 0/24 with it off).

Four hours later (2026-09-09 20:48), #364 (8130041, `tied_correctness`)
merged. Its diff changed `decode_graph=not args.deterministic` back to
`decode_graph=True` and deleted the `--deterministic` argument. The PR was
about a binary correctness tie fraction; the flag removal was not in its
body, review, or comments. CI was green on both checks. The flag existed on
`main` for under five hours.

## Root Cause

**Surface:** #364's branch was based on `5cd9404` — #357's own merge commit.
The branch could see the `--deterministic` lines, and its diff deleted them.
The deletion is visible in #364's own diff, unambiguous. There was no stale
branch, no silent three-way merge: the author deleted the lines, and the
merger approved without reading the diff.

**Why it was not caught:** the merger looked only at the two green checks
and did not open the diff. A PR that deletes a feature can be double-green,
CLEAN, no conflict. Green is not review.

**Deep:** `--deterministic` was mentioned in two docs pages, but only in
prose ("`--deterministic` is a working workaround"). The doc-link gate
(`test_no_doc_invokes_a_flag_the_cli_does_not_have`) is scoped to flags on
the same line after a `tilerl <sub>` invocation — a scope deliberately
narrowed from an earlier version that reported 18 false positives. A flag
that appears only in prose has no gate watching it. The merge removed it;
nothing turned red.

## The wrong attribution

The first version of this entry attributed the deletion to a stale-branch
merge: "#364's branch was based on a commit before #357 landed, and GitHub's
three-way merge took the branch's version silently." That mechanism was
invented. Three git commands were run to verify it, and each failed and
returned empty/0/error:

1. `git merge-base --is-ancestor 5cd9404 8130041^2` — `8130041` is a squash
   commit with one parent; `^2` does not exist. The command errored, and the
   `||` branch printed "did NOT contain."
2. `git show "$tree:src/tilerl/cli.py"` — printed the root directory listing,
   not the file. `grep -c deterministic` returned 0; `grep -c "def main"`
   also returned 0. No positive control was run.
3. A later read pasted a tree oid and path together (`fatal: Not a valid
   object name`), likewise returning 0.

All three failed in the same direction — "the line is not there" — so they
appeared to corroborate each other. Debunking needed one positive control:
the same read on a string that must exist (`def main`) also returned 0. A
negative grep needs a positive control; this was violated three times in one
hour.

The truth, found with commands that can fail:

```
$ git merge-base --is-ancestor 5cd9404 d0bb05af   # d0bb05af = #364 branch head
YES

$ git show d0bb05af -- src/tilerl/cli.py | grep '^-.*deterministic'
-                          decode_graph=not args.deterministic,
-    p_train.add_argument("--deterministic", action="store_true",
```

## Fix

Restore `decode_graph=not args.deterministic` and the `--deterministic`
argument (keeping #364's intentional `spec_depth=args.depth`). Add a wiring
test (`tests/test_deterministic_flag.py`) that parses the flag on/off and
asserts the source wiring — it goes red if the argument is removed or the
`not` is deleted. The process fix: read the diff before merging, not just
the CI status.

## Rule

1. **Green is not review.** A PR that deletes a feature can be double-green,
   CLEAN, no conflict. The merger must read the diff, not the status.
2. **A flag's existence is guaranteed by the test that wires it**, not by
   docs mentioning it. Docs say what it does; the test says it is still
   there. A flag that appears only in prose is invisible to every gate.
