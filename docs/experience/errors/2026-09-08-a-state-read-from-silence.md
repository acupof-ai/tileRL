# A state read from silence, and a list read from memory — 2026-09-08

**Status:** two sessions reported repository state wrongly within one exchange, in opposite
directions, both while holding the evidence. Neither error changed an outcome, which is why
they are worth writing down: nothing external would have caught them.

## The pair

**One session listed the day's merges into `main` from memory: four commits.** There were
five, and the missing one was `0e52f89` — precisely the merge that answered the question under
discussion (how #319's content had reached `main`).

**The other — me — reported an open PR as "green for a while now" when it had been merged 90
minutes earlier.** The proof was already in my own terminal twice: a `git log --oneline
origin/main -1` twenty minutes prior, and a `git rebase` that printed `12da5a0` in its output.
One `gh pr view 322 --json state` returned `MERGED`.

| | mechanism | what was available |
|---|---|---|
| the list | recited something that had been read | the commits, in `git log` |
| the status | inferred from the absence of a message | the merge sha, printed twice |

## The second one is the worse mechanism

Misremembering a list is a failure of recall about something observed. Reporting a PR as open
because no one announced merging it is **taking silence as a reading**, and silence is
permanently true in a multi-session repository: nobody is obliged to announce pressing a merge
button. So the inference has no failure mode — it returns "still open" whether or not that is
the case, and it feels like knowledge rather than like a guess.

**A default value dressed as an observation** — the same family as reading a blank field as a
measurement.

### It is a class, and four instances landed in one day

`tilerl-27` enumerated the rest from the same afternoon:

| belief | why it was believed | actual reason for the silence |
|---|---|---|
| #322 still open | nobody said they merged it | merging is not announced |
| two peers had not pushed | they had not said so | pushing is not announced |
| card 3 still occupied | no handover message | releasing a card is not announced |
| #323 had not triggered CI | the check list looked empty | the checks had not been queried |

**All four read a missing signal as a negative state**, and in every case the signal was
missing because nobody owns sending it — not because the event had not happened.

## What replaces both

Any cross-session state comes from one call, never from the inbox:

```sh
gh pr view <n> --json state,mergeCommit    # not "did anyone tell me"
gh run list --branch <name> --limit 3      # not "the checks look empty"
git log --oneline origin/main -5           # not a recited list
```

Plus the card claim table for card ownership. **An inbox answers "who spoke to me", not "what
is the state of the world."**

Each is one call. The cost of the wrong answer here was two corrections in one conversation;
the cost of the call is a second.

## The missing default view

One of the day's other findings has the same structure and a sharper fix. A PR's content is
`merge-base..head`, and **no step in the workflow shows that by default**: the author sees
their own commits, the reviewer sees GitHub's rendered diff (correct, but silent about whose
commits are in it). So a PR silently carrying another PR's three commits was nobody's
oversight — the view that would have shown it is not on anyone's screen.

A "remember to check" rule decays; a command does not:

```sh
git diff --stat $(git merge-base origin/main HEAD)..HEAD
```

Run it before opening a PR, and if it disagrees with the description you were about to write,
fix the description. It caught a discrepancy on the very PR that introduced the rule, which is
not luck: the reason it catches things is that nobody has looked at that diff, and that holds
for every PR including this one.

## The relay incentive, corrected

The same exchange produced a sharper account of why a relayed observation loses its hedge. The
first explanation was economic — a caveat costs words, so it gets trimmed. That is true and
too weak. **The stronger pull is that the trimmed version reads as more useful, not as
lazier.** Relaying "I did not read further" makes the relayer look unfinished; relaying "this
looks like a real failure" looks diagnosed. So the omission leaves no trace of laziness to
find in a self-check.

The executable form, from `tilerl-27`: **put the scope limit and the conclusion in the same
clause, so it cannot be trimmed as a suffix.** "They saw X but stopped before Y" is one
sentence, not two. And the signal to watch for: *if you cannot state the limitation without
looking as though you did not finish, that is the moment you are about to drop it.*

One case from the same day, in both forms. Two sentences: *"the denominator is Σlen"* plus
*"but it has not been checked on data where the two differ"* — the second is droppable, and
the first alone reads as settled. One sentence: *"on a saturated arm this figure satisfies both
definitions, so it cannot distinguish which one it is"* — the limit is the main clause. The
recipient later upgraded that to "unfalsifiable in this setup", which is a conclusion that can
only grow from the one-sentence form.

## A positive case, since the day produced only negative ones

The same rule has a demonstrated payoff, and it is worth recording in the direction that
worked. Two forms of one finding:

**Two sentences.** *"The denominator is Σlen."* / *"But it has not been checked on data where
the two differ."* The second is droppable, and the first alone reads as settled.

**One sentence.** *"On a saturated arm this figure satisfies both definitions, so it cannot
distinguish which one it is."* The limitation is the main clause.

The one-sentence version went out, and the recipient upgraded it to **"unfalsifiable in this
setup"** — a stronger and more useful statement than the original. That upgrade has no starting
point in the two-sentence version once the second sentence is trimmed. **Keeping a limit inside
the sentence is what gives the next person something to push forward from.**

## Rules

- **Read cross-session state; never infer it from your inbox.** `gh pr view --json state` for a
  PR, `gh run list --branch` for CI, the claim table for a card. An inbox answers "who spoke to
  me", not "what is the state of the world" — and a missing signal usually means nobody owns
  sending it, not that the event did not happen. Four instances in one afternoon.
- **An inference from silence has no failure mode**, which is exactly why it cannot be checked
  and why it feels like knowledge.
- **Recite nothing that a command can read.** A merge list from memory dropped exactly the
  element the question turned on.
- **Replace a "remember to check" rule with a command.** `git diff --stat $(git merge-base
  origin/main HEAD)..HEAD` before opening a PR; a habit decays, a copyable line does not.
- **Put a scope limit in the same clause as the conclusion.** A limitation appended as a second
  sentence is the part a relay drops, and it drops it because the shortened version reads as
  more competent rather than as lazier.
- **When you cannot state a caveat without appearing unfinished, that is the signal you are
  about to omit it.**
