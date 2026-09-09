# A result cache without a code version makes two same-config runs start from different states

Date: 2026-09-09
Status: open (fix is a policy decision: sha in the key, or refuse-to-train on a hit)

## Context

The before-arm eval (MMLU 1000 + GSM8K 500) is cached by content hash under
`runs/eval-cache/<key>.json` (per-tree; `cli.py:792`). The key
(`_before_eval_key`, `cli.py:360`) covers weights (path/size/mtime), sampling,
eval_n, concurrency — **not the code sha**. A hit skips the eval entirely:
zero requests, and the engine reaches step 1 clean. A miss runs all 1500
problems first, so step 1 runs on an engine that has already served 1500
decodes.

Three runs of the same recipe, seed 0:

| run | tree | cache | before-arm | step-1 engine | outcome |
|---|---|---|---|---|---|
| 86a06dc8c420 (bd72288) | v100-sm70-fp4 | **hit** | 0 requests | clean | survived to step 100 |
| 3276b687898d (2b55eb6) | realrun-ctl (fresh) | **miss** | 1500 requests | pre-warmed | collapsed at step 35 |
| check 3, same id (a93de2e) | eval25-ctl (fresh) | **miss** | 1500 requests | pre-warmed | collapsed at step 35 |

The surviving run's tree held the cache file from an earlier run the same day
(Sep 8 13:33); both collapsed runs were in fresh trees, where a miss is
guaranteed. Whether a run pre-warms its engine is decided by the **tree's
history**, not by any recorded config field — the manifests differ only in
`eval_before_cache.cache_hit`.

Worse: with one-run-one-worktree and the worktree cap, trees are constantly
created and deleted, so **worktree age decides the training trajectory** —
new tree → miss → pre-warmed engine; old tree → hit → clean engine. The
trajectory depends on a pure ops variable that appears in no manifest, no log
line, and no entry.

For a day the divergence was attributed to the sha gap between the runs. The
gap contains no training-computation change (every commit diffed: #324 is
timing/logs only, #328/#329/#330/#334/#336 touch eval plumbing and rollout
text, none the training math). The actual difference is cache state.

## Root cause

The cache key has no code-version dimension, and the cache lives in the tree.
A fresh tree always misses; a reused tree may hit. Same config, same seed,
different pre-step-1 state — and the log's only witness is one line
(`eval before: cache hit ...`).

Whether pre-warming *causes* the collapse is still open: the clean side has
n=1, so "pre-warmed collapses, clean survives" is consistent with the cache
story and with one clean run happening not to collapse. The A/B experiment
(clean vs pre-warmed, 3 steps) tests whether pre-warming changes the
trajectory at all; a full pre-warmed rerun of the survived config tests the
collapse directly. This entry is about the cache, not the collapse verdict.

## Fix

None shipped. The first three options below are mitigation: they change the
cache, but the cache is not the broken half. The cached value is legitimately
reusable — the before-arm measures the *base* policy on a fixed problem set,
which does not change with training code; keying on sha would only re-pay
~1661 s per commit and would not fix same-sha reruns (the second run still
hits, still starts clean, still diverges from the first).

The broken half is that **training's start state depends on whether eval ran**.
A correct implementation gives step 1 an engine independent of prior history:

- **(a) Unconditional fixed warmup before training** — same shape and count,
  cache hit or miss, so both paths converge to the same engine state.
- **(b) Separate engine instances for eval and training** (or rebuild/reset
  the engine between them). Priciest, cleanest semantics.

Mitigations, if (a)/(b) are too expensive right now:

1. Refuse to train on a cache hit when the sha differs from the one that paid
   it (record the paying sha in the cache file).
2. Add the sha to the key — simple, but silently lowers the hit rate and
   throws away the cache's payoff (see wins/2026-09-05-before-eval-cache.md).
3. Keep the key, but print the cache state in the run's header line, not a
   log line nobody greps.

## Rule

A result cache without a code version lets two "same-config" runs experience
different pre-states, and the log's only witness is one `cache hit` line. When
two same-config runs diverge, check cache state — in every tree they ran in —
before suspecting code. And when a run's start state depends on cache state,
it depends on worktree age: an ops variable recorded nowhere.
