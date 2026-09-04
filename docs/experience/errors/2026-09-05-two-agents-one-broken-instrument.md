# Two agents, one broken instrument, two opposite wrong conclusions — 2026-09-05

**Date:** 2026-09-05
**Task:** merging PR 67; recovering three commits it did not carry
**What broke:** `gh pr checks` reports the most recent CI run for a PR, not the run for
its current head. Two sessions read it, drew opposite conclusions, and neither corrected
the other.

## Context

`docs-disk-full` carried the V100 chat-UI work. I pushed three more commits
(`bd9509d`, `986af3b`, `0aa930d`) on top of the tested head `550740a`. The peer session
held merge authority and was waiting for CI.

Both of us then used `gh pr checks 67`:

| | read | concluded | acted |
|---|---|---|---|
| peer | `gate (macos-14) pass`, `gate (ubuntu-latest) pass` | the current head is green | squash-merged |
| me | same output, via a watch script printing `ALL CHECKS SETTLED` | CI re-ran on the new head and passed | told the peer "CI will re-run on 0aa930d" |

Both readings were wrong, in opposite directions, off identical output.

## Root cause

`gh pr checks` prints check names and conclusions with **no sha**. The green belonged to
`550740a`, one commit before my three. Queried by sha, the picture is unambiguous:

```
gh api repos/acupof-ai/tileRL/commits/550740a/check-runs  ->  3 checks, all success
gh api repos/acupof-ai/tileRL/commits/0aa930d/check-runs  ->  total_count = 0
```

Zero. Not pending — never triggered. `.github/workflows/ci.yml` fires on
`push: branches: [main]` and `pull_request: branches: [main]`, so a push to a feature
branch runs nothing, and the `pull_request` path needs the PR to update. GitHub's PR
index was lagging: the git ref API already returned `0aa930d` while the PR API still
reported head `550740a` and its commit list did not contain any of the three.

**So a state exists that neither of us had a name for: commits on a pushed branch that no
CI run has ever seen, while `gh pr checks` reports green.** Both failure directions come
from that state, and the command cannot express it.

## What saved it, and what did not

The merge landed `550740a` — the one sha that *was* tested — so nothing untested reached
main. That was the stuck index pinning the head, not a check doing its job. The peer's
own summary: *"my process was wrong and the result was right, and those are two things to
record separately."* With the index working, the same sequence merges untested code.

The three commits were not lost: they were still on the branch, absent from main
(`merge-base --is-ancestor` says so for all three, and main has no
`tests/test_support_matrix.py`). Recovered by branching from the new main and
cherry-picking.

## Fix

```
gh api repos/<owner>/<repo>/commits/<sha>/check-runs     # the sha's OWN checks
gh run list --json headSha,conclusion                    # then COMPARE the sha string
```

The second one still needs care: the newest run is not necessarily the current head's.
Here the newest run belonged to `550740a` while the ref was already at `0aa930d`.

A watch script must key on the sha it fetched, not on the PR:

```bash
SHA=$(gh pr view 71 --json headRefOid --jq '.headRefOid')
gh api "repos/.../commits/$SHA/check-runs"
```

Verified on the replacement PR before handing it over: head `9fcc989`, three check-runs
on that exact sha.

### The same defect, second form, twenty minutes after writing the rule above

That sha-keyed script then reported `ALL 1 CHECKS SETTLED` on head `32e6caa` — and it was
arithmetically right. Checks register over several seconds, so immediately after a push
only GitGuardian exists, and "every registered check is complete" is already true with
`total_count = 1`. Both gates were still unregistered.

So the second version failed the same way as the first: **the number it read was correct,
and it was the answer to a different question.** `gh pr checks` answers "was something
recently green" when asked "is this code green". Counting completed checks answers "is the
currently-known set finished" when asked "did CI pass". Keying on the sha fixed the first
substitution and left the second one untouched.

Fixed by naming the checks that must be present rather than counting whatever is:

```bash
for w in "gate (macos-14)" "gate (ubuntu-latest)"; do
  jq -e --arg n "$w" '[.check_runs[] | select(.name==$n and .status=="completed")] | length > 0'
done
```

## Rule

**A green is a claim about a commit, so the check must name the commit.** Any CI query
that returns conclusions without a sha cannot answer "has *this* code been tested" — it
answers "was something recently green".

**Assert on the checks that must be present, never on a count of what is present.** A
count is a claim about the set the API happens to know right now, and that set grows for
several seconds after a push. Both failures in this entry are the same substitution: a
number that is arithmetically correct and answers a question nobody asked. The fix in both
cases is to name the thing — the sha, then the gates — rather than to trust a quantity
derived from whatever was in scope.

**Cross-checking is worth what the instruments' independence is worth, not what the head
count is.** Every other mutual check between these two sessions today worked — the peer
reports three wrong instructions of theirs stopped by mine, and my own non-existent
single-quote vulnerability was retracted after their review — and each time the two of us
had looked through *different* instruments. This once we used the same command, and the
redundancy went to zero while feeling unchanged. Two agents agreeing is not evidence; two
methods agreeing is ([agreement-is-not-verification], now with a measured instance).

**Distinguish "the result was right" from "the process was right" out loud.** The merge
was correct by accident. Recording it as a success would have kept a broken check in the
flow with a green next to it.

**Writing the rule down does not install it.** The second form above happened twenty
minutes after this entry's first draft, in the script written to obey it, by the author of
both. A rule stops a defect only where a check enforces it
([guard-at-point-of-use-not-memory]) — here that meant the gate names in the loop
condition, not the sentence in this file.

