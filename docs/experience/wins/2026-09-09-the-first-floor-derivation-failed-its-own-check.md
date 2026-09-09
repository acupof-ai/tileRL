# The first floor derivation failed its own check — and the first reader caught it

## Context

The ruler system (PR #350) made `floor.derivation` a required field: every
measurement carries a floor with a computation anyone can recompute. The
first derivation written under that rule was the decode roofline:

```
129 tok/s = 30.9 GiB weights / 4 TB/s H20 HBM
```

27 recomputed it while reviewing the PR: 30.9 **GiB** / 4 TB/s = 8.295 ms =
**120.6** tok/s, not 129. The source entry
(`wins/2026-08-24-sota-all-levers.md`) says 30.9 **GB** — 30.9 GB / 4 TB/s =
7.725 ms = **129.4** tok/s. The number was right; the unit label was wrong.

## What worked

The defect was a unit label in a string, and it was caught in review, before
merge, by arithmetic. In the old world the same string lived inside a
`print()` in a probe script; nobody recomputes print output, and the error
would have lived forever. The derivation field exists so that a reader can
recompute the floor — the first reader did, and the field failed its own
check on its first use. That is the system working as designed: the
derivation is the only field the machine cannot validate, so the machine
makes it cheap for a human to.

## Rule

Every new `floor.derivation` is recomputed by its reviewer — the arithmetic,
not just the prose. A derivation nobody recomputes is a comment.
