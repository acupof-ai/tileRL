# A sha confirmed, and what the sha contained was not

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55
**Status:** closed — the run was restarted from `fd0c876` before any curve point landed

## Context

A 100-step GRPO run was launched on the H20 to measure `steps_to_score` on GSM8K, with four
curve points at steps 25/50/75/100. Before launching, three things were checked and written
into a pre-registration: the sha (`12da5a0`), the full flag set, and the pre-registered
criteria — including the saturation threshold, which is **1.00 pt paired** (McNemar at 5%
discordant) rather than the 1.90 pt unpaired width at n=500.

The paired threshold is only computable from per-problem rows on disk. Those rows, and the
`mean_len` / `at_cap` fields that separate a real score from a truncation artefact, are added
by **#323 — which was still OPEN**. The run had been launched from main, and main did not
have them.

Caught at step 24 of 100, about 40 minutes in, by `tilerl-27` asking one question: does the
code that is running contain #323?

## Root cause

Three verifications ran before the launch and all three passed. Each verified a different
property, and none of them verified the one that mattered:

| checked | what it establishes |
|---|---|
| sha `12da5a0` | which commit the tree was synced from |
| the flag set | which arguments the process received |
| the criteria table | which numbers would decide the verdict |
| **the code contains the fields the criteria need** | **never checked** |

The pre-registration document made the gap harder to see, not easier: it stated the paired
1.00 pt threshold, cited `_write_eval_rows`, and named the three fields — so it read as though
the mechanism were present. **A document describing a mechanism and a tree containing it are
independent facts**, and the document was written by the same session that assumed the tree.

The near-miss that made it recoverable: my own report to the peer contained the sentence "this
is the quantity `eval_secs` was added to record (#323), and it lands with the first curve point
at step 25." That sentence is **true under both readings** — the code has #323, or the code
will have it once #323 merges — so the peer correctly refused to infer from it. It also
contained a second error: `eval_secs` is from #309, not #323. The attribution was wrong and the
ambiguity was what saved it.

## The reading that settled it

Not the PR state, and not the sha. The code:

```
/work/tilerl-s-v100-sm70-fp4/src/tilerl/cli.py:831
    curve.append({"step": step, "correct": c, "total": n, "score": c / max(n, 1),
                  "secs": ..., "eval_secs": ..., "jit": not curve})
grep -n "at_cap" → no matches
```

Plus two independent confirmations that this file is the one executing: `/proc/1150334/cwd`
resolves to that tree, and `python3 -c "import tilerl.cli; print(__file__)"` from that tree
returns that path. `.synced_commit` reading `12da5a0` is consistent but weaker — it records
what was synced, not what is loaded.

`gh pr view 323 --json state` returning OPEN is a **sufficient** argument (an open PR is not in
main, so a tree synced from main lacks it) and it is the argument the peer used. It is still
one inference removed from the artefact: it reasons from the merge state to the file contents.
Reading `curve.append` reasons from nothing.

## Fix

Restarted from `fd0c876` (main with #323). Cost ~15 minutes rather than the ~40 first
estimated, and the difference is worth recording because it decided whether to restart at all:

- the **before-arm eval is a cache hit** — `runs/eval-cache/68256c15….json` was already on
  disk, its key covers weights/config/eval_file/eval_n/matcher/sampling (`cli.py:344-353`) and
  none of those change across the restart, and `pod_sync.sh:49`'s wipe is
  `find . -mindepth 1 \! -path './runs' \! -path './runs/*' -delete`, so the cache survives a
  re-sync. That arm is ~20 of the ~25 minutes of startup.
- the 24 completed steps re-run at 23.3 s each.

My first estimate of the restart cost was the peer's 40 minutes taken at face value, which
counted the before-arm as re-run. **A cost estimate for redoing work has to ask which parts are
memoized**, and here one directory being excluded from a wipe was the whole difference.

**And I said those 24 steps would replay identically. They did not** — that claim went into two
documents before it was checked, and checking it took one command. Both logs at the same step
numbers:

| steps | rollouts (`tok`, `reward`) | `ce` |
|---|---|---|
| 1-3 | **identical** — 250/256/248 tokens, rewards 0.75/0.00/0.125 | **differs** — step 1 is 1.5197 vs 1.5380 |
| 4 onward | diverged — step 4 is 116 vs 128 tokens, reward 0.875 vs 1.000 | diverged |

`--seed 0` fixes the prompt order (`train.py:481`, no shuffle) and the rollout seeds
(`seed + step*group + g`), which is why the first rollouts are the same tokens. It does not make
the training arithmetic reproducible across processes, and one differing optimizer step
separates the trajectories. The code is not the cause: `git diff 12da5a0..fd0c876 -- src`
touches `cli.py`'s logging and curve fields and `recipes.py` comments — nothing in `train.py`,
`model.py` or `backend.py`.

**The mechanism is not pinned, and stating a candidate is not pinning it.** One verified
non-code difference: the first attempt's before-arm was a cache *miss* and ran MMLU 1000 +
GSM8K 500 on the card before step 1 (`curve.log` has the `gsm8k greedy` and `mmlu 0-shot`
lines, `curve2.log` has neither), so the processes entered training with different allocator
histories. Whether that reaches the arithmetic is untested; ordinary fp4 reduction
non-determinism is the competing explanation. Distinguishing them needs a same-process repeat,
which nothing currently planned requires — so it stays a candidate, and the reproducibility
claim stays withdrawn rather than replaced.

Verified after the re-sync, before relaunching: `.synced_commit` = `fd0c876`, `grep -c at_cap`
= 4, the cache file still present. After the launch, the field names in the running
`curve.append` (`cli.py:872`, `mean_len` and `at_cap` present) — not the sha.

## Rule

**Verify that the code contains the mechanism, not that the tree is at a revision.** A sha
answers "which commit"; it does not answer "does this contain the feature my criteria need",
and when the feature is in an unmerged PR the two answers differ. Read the construct — the
dict literal, the function, the field — from the file the running process loaded.

Two corollaries from how this one was caught:

- **A statement true under both readings cannot confirm either.** "This is the quantity #323
  records" is compatible with the code having it and with the code lacking it. When a peer
  asks a yes/no question about state, answer with the reading, not with a sentence that
  mentions the mechanism.
- **A pre-registration is evidence about intent, never about the tree.** It named the three
  fields and cited the function that writes the rows, which is exactly why it did not read as
  a gap. The document was the strongest-looking evidence and it was not evidence at all.
- **"Same seed" is a claim about inputs, not about outputs.** I wrote "the 24 steps replay
  identically" into two documents from the seeding code alone. The rollouts did match and the
  loss did not, and one `diff` of the two logs would have said so before either document
  claimed it. A determinism claim needs the two outputs compared, not the seed read.

Same family as [green checks that proved less than they looked](2026-09-02-green-checks-that-proved-less-than-they-looked.md):
a claim about code settled from metadata *about* the code rather than the code.
