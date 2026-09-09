# A flag reverted by a stale branch — `--deterministic` silently removed

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

**Surface:** #364's branch was based on a commit before #357 landed. The
author edited the same line (`spec_depth=args.depth, decode_graph=...`) on
the old base. GitHub's three-way merge took the branch's version silently —
no conflict, no warning, CLEAN. #357's change was overwritten by a PR that
never intended to touch it.

**Deep:** `--deterministic` was mentioned in two docs pages, but only in
prose ("`--deterministic` is a working workaround"). The doc-link gate
(`test_no_doc_invokes_a_flag_the_cli_does_not_have`) is scoped to flags on
the same line after a `tilerl <sub>` invocation — a scope deliberately
narrowed from an earlier version that reported 18 false positives. A flag
that appears only in prose has no gate watching it. The merge removed it;
nothing turned red.

## Fix

Restore `decode_graph=not args.deterministic` and the `--deterministic`
argument (keeping #364's intentional `spec_depth=args.depth`). Add a wiring
test (`tests/test_deterministic_flag.py`) that parses the flag on/off and
asserts the source wiring — it goes red if the argument is removed or the
`not` is deleted. The merge gap is a process fix: a PR that touches `src/`
and is based on a stale `main` rebase before merge.

## Rule

A flag's existence is guaranteed by the test that wires it, not by docs
mentioning it. Docs say what it does; the test says it is still there. A
flag that appears only in prose is invisible to every gate.
