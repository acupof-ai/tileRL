# A run that could not be replayed — 2026-09-09

**Status:** the day's headline result — `steps_to_score(X=91.0) ≤5 steps / ≤119 s`,
a ≥17.4x lower bound on the full 2066.7 s run — came with a −11.0 pt collapse at
step 75 that no gate saw. Asked to read what the policy wrote across that window,
the answer was: **the text was never saved. No run in this tree has ever saved it.**

## Context

Run `86a06dc8c420` (GRPO, 27B, GSM8K, 100 steps) produced the steps-to-score
curve. Step 50 scored 93.4%, step 75 scored 82.4% (paired −11.0 pt, 6.62σ),
step 100 recovered to 91.2%. MMLU held (75.1→75.7), so general capability was
intact; 0/500 cap hits, so it was not truncation; train reward held
(0.900→0.855), so it was not training divergence. The remaining question was
what the policy *became* on GSM8K text — format change, degeneration, or a
length effect — and the task was to read ~20 completions each from steps
26-50 and 51-75.

## Root cause

`train.py:578-582` wrote one row per completion with fields
`step/p/g/tokens/reward/advantage`. The completion token ids existed in memory
(`comps`) and were dropped at the write site. The eval files saved even less:
`i/correct/tokens/answer`, where `answer` is the gold, not the model output
(MMLU rows carry a `prediction`, but only the letter). The run directory holds
nine files; **none of them contains a single token of model output.** Only the
final step-100 adapter survived, so intermediate policies cannot be replayed
either.

This is not data lost. It is data the instrument was built not to keep — the
same family as a citation to an artifact the tree lacks: a question the
apparatus cannot answer, discovered when the answer mattered.

## Fix

`grpo_loop` takes a `decode` callable; when given, each rollout row carries
`text` (the decoded completion). The CLI's training path passes `tok.decode`.
Cost is ~0.5 MB per 100-step run (150 tok × 8 × 100), noted at the write site
so nobody adds a switch for it. `tests/test_ledger.py` asserts every row of a
2-step run has non-empty `text`; the negative control (the field deleted) goes
red, so the gate exercises the field.

## Two findings the numbers do support

**Length moved in opposite directions on the two populations.** Training
rollouts (temperature 1.0, cap 256) *lengthened* across the collapse window —
135.6 → 156.9 mean tokens, cap hits 8/200 → 14/200 — while greedy eval answers
(cap 2048) *shortened* (153.1 → 122.8). These are parallel observations of two
populations, not two measurements of one quantity. The coherent reading: the
policy's distribution **widened** — sampling runs longer and hits the cap more
often, while the greedy mode gets shorter. A mean alone would have hidden the
shape change.

**The collapse has no training-side echo at all.** On the exact step where
eval lost 11 points, train reward was 1.000; the window reward dipped only 5%.
Training-side metrics — reward, length, the six gates — gave no signal that
could have stopped the run before step 75. This is the strongest evidence in
the tree that **training-side metrics cannot be the early-stop signal**; the
stop decision has to live on the eval curve, which is what the steps-to-score
instrument now prices.

## Rules

- **A run that can change a conclusion must save the raw material the
  conclusion is made of.** Tokens/reward rows answer "how long and how right";
  they cannot answer "what did it write". When a collapse is first noticed is
  too late to decide to save text.
- **A negative control belongs in the same change as the gate.** The test
  asserts non-empty `text`; deleting the field makes it fail. A gate that
  cannot go red on the defect it guards is decoration.
- **Two populations moving oppositely is a shape finding, not a mean finding.**
  Report both populations and the direction of each; do not average them into
  one "length changed" sentence.
