# A diagnostic that returned something shaped like an answer — 2026-09-08

**Status:** three instrument failures in one 20-minute diagnosis, all of them permissive. The
PR involved (#319) turned out to need closing rather than fixing.

## What happened

`tilerl-27` relayed that #319's two CI gates were red, with two lines from the log:

```
sh: NameError
assert len(moved) == len(two_d)
```

and the judgement *"this does not look inherited"*. `assert len(moved) == len(two_d)` lives in
`tests/test_iso.py`, which another session had been reading that day, so the working
hypothesis was an interaction between two branches.

None of that was true. The actual failure, from the full job log:

```
FAILED tests/test_ledger.py::test_train_cli_writes_manifest_and_is_idempotent
  - RuntimeError: PagedKvPool exhausted: all 137 blocks in use
= 1 failed, 504 passed, 20 skipped, 6 xfailed in 1106.35s
```

That is the pool-sizing defect fixed the same afternoon in #320 and #322 — inherited from a
4-commit-stale base. #319 touches `CHANGELOG.md`, two `docs/experience/` entries and
`scripts/prof_grpo_step.py`: **no test file, so it cannot have broken one.** That check alone
would have settled it before any log was read.

## Failure 1: `gh run view --log-failed` returns the format diff

On this workflow, `--log-failed` returns the output of the `ruff format --check` step, whose
diff is full of source fragments:

```
- Scale: T.Tensor((N, K // block), "float16" if sh else "float32")   # sh: NameError
54 | assert len(moved) == len(two_d)
```

Those are lines of **our own code being reformatted**, complete with a comment containing the
word `NameError` and an `assert` from a test file. They read exactly like pytest output. Two
passes over the log were needed before the real `short test summary` appeared.

The correct invocation:

```sh
gh run view <run-id>                       # which STEP failed
gh run view <run-id> --job <job-id> --log  # then find "short test summary info"
```

`--log-failed` is the flag whose name promises the answer, and here it returns a plausible
wrong one. Same family as the rest of today: an instrument answering an adjacent question
cleanly.

## Failure 2: a hedge does not survive relay

The originating session's own words were that it *had not read further*. What arrived was
"this does not look inherited" — a conclusion. The observation was explicitly marked
incomplete, and the marking is what got dropped.

`tilerl-27` identified the mechanism: **a relay's cheapest addition is certainty, because the
hesitation in the original costs words to carry and the conclusion does not.** Second instance
that day; the first was a PR head's green read as main's green.

## Failure 3: the PR's content was not its description

After rebasing onto current `main`:

```
$ git rebase origin/main
dropping 402762b ... -- patch contents already upstream
$ git diff --stat origin/main..HEAD
 CHANGELOG.md | 2 ++
```

Both entry files were already in `main`, and so were both CHANGELOG lines
(`git show origin/main:CHANGELOG.md | grep -c` → 2). Cause:

```sh
git merge-base --is-ancestor 402762b origin/fix/eval-arm-sizes-the-pool   # true
```

#320 was branched from #319's head, so merging #320 landed #319's three commits with it.
**Neither PR description said so**, and re-pushing #319 would have printed the same verdict
into the CHANGELOG twice.

Nobody was careless; nobody had run the one command that shows what a PR contains:

```sh
git diff --stat $(git merge-base origin/main HEAD)..HEAD
```

This is the authoring-side twin of a CI result describing a head rather than a merge (see
[a disjunction reasoned one term at a time](2026-09-08-a-disjunction-reasoned-one-term-at-a-time.md)):
there, the green belongs to a commit that is not what `main` becomes; here, the diff belongs
to a range that is not what the description claims.

## A related list read from memory

In the same exchange, the day's merges into `main` were listed as four commits. There were
five, and the missing one was `0e52f89` — precisely the merge that carried #319's content in.
A list recited rather than read, missing the one element that mattered to the question being
asked.

## Rules

- **Ask what the change touches before reading its failure log.** A diff with no test file
  cannot break a test; that comparison is cheaper than any log and settles the inherited-vs-own
  question outright.
- **`gh run view --log-failed` is not the failure on a multi-step workflow.** Use
  `gh run view <id>` to find the failing step, then `--job <id> --log`, then search for
  `short test summary info`.
- **Carry the hedge with the observation.** "I did not read further" is part of the finding;
  dropping it converts an unfinished look into a conclusion, and relays drop it because
  certainty is shorter than doubt.
- **A PR contains `merge-base..head`, not what its description says.** Run
  `git diff --stat $(git merge-base origin/main HEAD)..HEAD` before opening one, and if it
  differs from what you were about to write, fix the description.
- **Recite nothing that can be read.** A merge list from memory dropped the single commit
  that answered the question.
- **Closing a redundant PR: never `--delete-branch`.** It closes any stacked child PR, and a
  closed PR can be neither reopened nor rebased onto a new base.
