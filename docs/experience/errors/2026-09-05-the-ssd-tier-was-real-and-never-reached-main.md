# The SSD tier was real, and the fix that removed the claim said it never existed — 2026-09-05

**Date:** 2026-09-05
**Arch:** target-independent (git history and a served string)
**Task:** correcting the stated reason for a merged fix

## Context

`#75` removed a claim from the served landing page: **"Paged KV + SSD tier —
prefix cache spills below HBM, reload skips prefill."** The removal was
correct. The reason given for it was not.

The commit message said:

> No such tier exists: `kv_cache.py:6` says "no preempt/swap, no cpu-offload",
> `PrefixStore` retains in-memory blocks, and the string SSD appears nowhere
> else in `src/` or `packages/`.

Every one of those observations is true **of the tree that was searched**, and
the conclusion drawn from them — that the tier never existed — does not follow.
A tiered prefix cache was written, reviewed and iterated on for four days:

| commit | date | what |
|---|---|---|
| `e9d5852` | 2026-08-29 | `feat(kv): HBM->SSD tiered prefix cache for prefill reuse` |
| `9a685d6` | 2026-08-30 | 6 review blockers + 2 nits against it |
| `29b58a3` | 2026-08-31 | `feat(kv): size-based LRU for SSD tier + graceful cold-hit degradation` |
| `3e53893` | 2026-08-31 | prefix-store capacity derived from HBM, not an entry count |

`KvTier` lived in `kv_cache.py` and `engine.py` across all four.

## Measured

The correction I first wrote for this was **also wrong**, and so was the peer's
reading I had accepted: we both said the tier reached `main` and was later
deleted, naming `a317f61` as the deletion. It is not a deletion — `KvTier` is
still present in `a317f61`'s tree.

Checked exhaustively rather than by picking commits. `KvTier` against every one
of the **567 first-parent commits of `main`**:

```
first-parent commits: 567
(no commit on main's first-parent history contains KvTier in src/)
```

And against the four commits that carry it:

```
e9d5852  ancestor-of-main = NO
29b58a3  ancestor-of-main = NO
a317f61  ancestor-of-main = NO
e4aaf8c  ancestor-of-main = YES   <- and has 0 occurrences of KvTier
```

So the real history is neither "it never existed" nor "it was merged then
deleted":

**`KvTier` was written on the sm70 branch and never reached `main`.** `e4aaf8c`
(`#60`) is the commit that brought that branch's server work over, and it
carried the landing page's two `SSD` strings **without the code behind them** —
`a317f61` has `SSD` twice in `server.py` and `KvTier` four times in
`kv_cache.py`; `e4aaf8c` has the same two strings in `server.py` and zero
`KvTier` anywhere.

That is the actual defect: a partial port left the advertisement and dropped
the feature. `#75` removed the advertisement, which was the right end to fix.

## Why the wrong reason mattered

"No such tier exists" reads as *this was never built*, which makes the string
look like someone describing an intention. The truth is that it **was** built,
and a port dropped it — which is a different class of problem and a different
prevention. Someone reading the `#75` message and wanting the feature would
conclude they must write it; in fact there is a four-commit implementation with
a review pass on it, one branch away.

## What the port left on `main`

Found by the peer independently checking this correction: `git log -S KvTier
--first-parent origin/main` returns **one** commit, `e4aaf8c` — which looks like
it contradicts the 567/0 scan. It does not. `-S` records *a change in how often
the string appears*, and in `e4aaf8c` every occurrence is in prose:

```
origin/main:CHANGELOG.md                                        3
origin/main:docs/experience/errors/2026-09-03-...premise-is-false.md   3
origin/main:docs/experience/wins/2026-08-31-sm70-gdn-chunk-fused.md    1
origin/main:docs/serve-v100.md                                  1
origin/main:src/ packages/                                      0
```

So `main` carries **eight mentions of a class it has never contained**, none of
them in code. The port brought the documentation and left the implementation,
and the landing-page string `#75` removed was one surface of that, not the whole
of it.

`docs/serve-v100.md:163` was the live one. It listed `--kv-tier <dir>` as the
third mitigation for a KV pool that does not fit 32K, said "the engine path is
tested", and offered it next to two real levers — while neither the flag nor
`KvTier` exists in `src/` or `packages/`. Anyone hitting that capacity wall would
have tried a flag `serve` does not have. Rewritten here to say the code was
written on a branch and never ported.

The remaining mentions are historical records — CHANGELOG lines and error
entries describing work as it happened — and those stay. The distinction is
tense: a record of what was done on some branch is true; an instruction to use
something is a claim about `main`.

## Rule

**A grep over the current tree answers "is it here now", never "did it ever
exist".** The evidence `#75` cited was a search of one tree, and the conclusion
was about all of history. When removing a claim as false, `git log -S` for the
thing being claimed costs one command and either finds the implementation or
raises the confidence.

**A `-S` hit is not a tree containing the string.** `git log -S` fires on
deletions and on prose, and it does not distinguish `src/` from `docs/`.
"Does this tree have it, and where" is `git grep` with a pathspec — a second
command, not an interpretation of the first.

**"Ancestor of main" is the question, not "is this commit real".** Both wrong
versions of this history named real commits with real dates and real subject
lines. `git merge-base --is-ancestor` separates work that shipped from work that
was written — the commits look identical without it.
