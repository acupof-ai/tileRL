# A fix argued twice, killed by 2 of 266, and three instruments that each returned a clean wrong table — 2026-09-08

> Status: **closed, nothing to fix.** The `_last_prefill_boundary(n)` signature defect is real —
> the boundary is `f(n, budget)` and the helper never sees the budget — and its cost on the live
> path is approximately zero. Two sessions independently argued the fix was worth doing, on two
> different grounds, and both grounds were wrong. What the round produced instead is three
> instrument failures and one reasoning failure worth more than the fix.

## Context

[The boundary entry](2026-09-08-a-one-token-chunk-made-last-unreachable.md) stays open on the
signature: `_last_prefill_boundary` takes `n` alone, so at n=961 it says 944 at budget 512 and
**448** at 511. The publish gate is

```python
last = pf.prefill_from == _last_prefill_boundary(len(pf.tokens))
...
if pf.interior_published == 1 or last:
    self._publish_prefix(pf, pf.prefill_from, spill=last)
```

The `spill=last` half feeds the SSD tier, which is off by default and under an expiring reject
([the wrong-layer entry](2026-09-08-the-boundary-spill-measured-at-the-wrong-layer.md)). So the
argument for fixing the signature had to come from the other half — the publish itself, which is
matchable in HBM and through the DRAM tier, both live.

Two arguments were made for doing it:

- **Mine:** the signature is wrong on its own terms, so it is a different defect from the spill
  fix that was just rejected.
- **`tilerl-27`'s, explicitly stronger:** "wrong on its own terms" is also what a fix feeding a
  dead path looks like, so check the consumer. `last` gates a publish, not only a spill, and a
  wrong `last` costs a match-path publish.

## Both arguments are wrong, and the reason is one word in the gate

`interior_published == 1 or **last**`. The first interior boundary publishes **unconditionally**.
So a wrong `last` costs a publish only where the deepest interior boundary is a *different
position* from the first one. Neither argument enumerated that.

Driven through the real `_build_plan` — not a replay — over unaligned lengths 65..2048, stepping
by 7 so the tail varies:

| budget | unaligned lengths | `lb` not a chunk end | **publish actually lost** | mean depth lost |
|---:|---:|---:|---:|---:|
| 512 | 266 | **0** | 0 | — |
| 511 | 266 | 2 | 0 | — |
| 508 | 266 | 3 | 0 | — |
| 504 | 266 | 4 | **2** | 1008 |

Two lengths out of 266, both at budget 504, and on each of them the first interior boundary still
published — what is lost is a second entry. `budget == 512` is clean because c14511b closed it.

One structural fact makes the surface smaller than it looks: **block-aligned lengths never reach
`last` at all.** `_last_prefill_boundary` returns 0 when `not tail`, commented "the
prompt-complete branch handles it," and `_finish_prefills`'s DONE branch publishes them. By
design.

## The reasoning failure: a disjunction reasoned one term at a time

27 misread this same expression **twice in one evening, in opposite directions**:

| | which disjunct was reasoned about | what was missed | wrong conclusion |
|---|---|---|---|
| earlier, on #271's "publishing ceiling" | `interior_published == 1` | `or last` publishes the deep entry too | a ceiling that does not exist (withdrawn as reading 1 of [the four-mechanisms entry](2026-09-08-four-mechanisms-one-regression-and-a-copied-flag.md)) |
| this round | `last` | `interior_published == 1` absorbs its failures | a wrong `last` costs a match publish |

Same line, both halves, both times the disjunct under consideration was treated as the whole
condition. A disjunction cannot be reasoned about one term at a time — the other term may already
cover every case the first one fails on. It has to be enumerated over the inputs.

## Three instruments, three clean wrong tables

Every one of these produced a table that looked finished. None of them was noticed by looking at
the table.

**1. A hand replay of `_build_plan` that dropped a condition.** The real back-off is
`end == n and tail and short > 0`; the replay omitted `tail`, so it attempted the back-off on
every chunk that ended at `n`. It reported `n=513 → [496, 512, 513]` where the planner gives
`[496, 513]`. A third-hand transcription of a spec disagreeing with the spec — and the entry being
worked from carries the Rule *a predicate that predicts another function's behaviour needs a test
that runs both*. The rule was in hand and was applied to the engine's code, not to the probe.

**2. A failure table from before its own fix, read as current behaviour.** The
`chunks end at [64, 65]` rows sit under **## The failure** and are pre-c14511b. Post-fix the
planner gives `[48, 65]` and `lb=48` *is* a chunk end. Four lengths were reported as MISMATCH
when what they showed was the fix working. The stale source was a file written one round earlier
in the same session: **provenance inside your own output is not automatically fresh.**

**3. A sampling grid aligned with the structure being measured.** Stepping `n` by 16 hits only
block-aligned lengths, where `lb` is 0 by design and never equals a chunk end — producing
**117 of 117 "missing"**, a 100% defect rate against a true rate of 0. Step 7 (coprime with
`BLOCK_TOKENS`) showed the real distribution. Same shape as reading a ramp as a step: the grid
was commensurate with the thing it was sampling.

## Rule

**Enumerate a disjunction over its inputs; do not reason about one disjunct.** Two wrong
conclusions came from the same line by considering each half separately. The other half is
exactly where the cases you care about may already be handled.

**A probe is a predicate too.** The rule about predicates that model another function applies to
the instrument, not only to the code under test. Drive the real function; a replay is a second
implementation and inherits nothing.

**A fixed entry's failure table is not current state.** It records what was wrong, deliberately.
Re-running against it reports the fix as a mismatch — and the risk is highest for a file you
wrote yourself, because your own output feels current.

**Choose a sampling step coprime with any period in the system.** A step of 16 against
`BLOCK_TOKENS = 16` measured one residue class and reported it as the population. Density does
not fix this and that is what makes it worse than a sparse grid: adding points at the same
stride stays inside the same residue class, so only changing the stride helps. A sparse grid on
mixed residues is caught by one extra point; a commensurate grid is caught by none.

**"How big is this branch" has four tools and they gave four different answers on one night.**
The question is almost always *what happens when this merges*, and three of the four answer
something else. Measured on two real PRs this session:

| | #291's branch (base merged) | #283's branch (already merged) |
|---|---|---|
| two-dot `git diff main branch` | +122 and **1000+ deletions** | +53, **−921** |
| three-dot `git diff main...branch` | **+310** (195 of it the parent PR's) | **+344**, −0 |
| `git merge-base --is-ancestor` | not on main | **"not on main"** — but it is |
| `git merge-tree --write-tree` | **+122, −0** | **empty diff** |

Two-dot compares against a main that moves, so every commit main gains reads as a deletion in
your branch — that is how a healthy branch looked like it would roll back a thousand lines of
other people's work. Three-dot compares against a fixed merge base, so a stacked branch whose
parent already landed counts the parent's content as new. `--is-ancestor` answers "is this
commit reachable", and **under a squash merge a branch head is never an ancestor of main**, so
it answers "not merged" for every squash-merged PR — feed it `mergeCommit.oid`, never a branch
head. Only `merge-tree` builds the merge and diffs that, which is why it needs no rule per
history shape: squash, rebase, stacked base all get the same right answer.

**And `merge-tree`'s conflicts are evidence, not failure.** The strongest use of it this session
was the reverse direction of the table above — content that looked like *redundancy* and was
not. A six-day-old stash was about to be dropped: both files it "added" already existed in the
tree, and a two-dot diff against it produced 1630 lines of unreadable CHANGELOG churn. Two
signals, both pointing at delete. `merge-tree` returned `CONFLICT (content):
tests/test_weights.py` — and a conflict means the stash holds something main does not, which is
the one fact that decides it. Reading that conflict as a failed merge would have destroyed 213
lines of someone's reasoning.

The credit is `tilerl-27`'s: it named `merge-tree --write-tree` and the reason — three-dot
counts from the fork point, so a stacked base landing double-counts. Two sessions had already
been bitten by two-dot in the same week, in opposite directions.

**When two independent arguments for a fix are both wrong, stop rather than find a third.** Two
data points in one evening that this area's intuition is biased toward "worth fixing" is enough
to hand the next item to someone else.
