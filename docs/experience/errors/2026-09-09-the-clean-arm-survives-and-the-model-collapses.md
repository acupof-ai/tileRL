# The clean-arm 100-step run survives — and the model collapses — 2026-09-09

## Context

#92 was a clean-arm 100-step GRPO run on the H20 pod (card 1, run
`8ae3e5ff6a95`): current sha, `grpo-gsm8k-27b`, `--max-new-tokens 512`,
`--eval-mmlu 0`, no `--eval-gsm8k`. The criterion was survival to 100
steps, not score. The run survived. The model collapsed.

## What happened

| metric | first | last | ratio |
|---|---|---|---|
| reward | 0.8295 | 0.1889 | 4.4x down |
| tokens/rollout | 104.9 | 5.47 | 19x down |

By step 87, rollouts were 0–3 tokens. The model learned to emit
end-of-sequence immediately, collecting the length penalty and avoiding
the risk of a wrong answer. `reward_rises=FAIL` (0.1889 < 0.8295).
`ce_last=8.957`. `tied_group_fraction=0.17` (83% of groups had at least
one member that differed, but the differences were 0–3 tokens of
silence).

`secs_per_step_median=11.68`, `peak_gib=43.37`, `steps_completed=100`,
`secs_total=1374`.

## Why

The recipe has no KL penalty and no length floor beyond the
`_refuse_short_rollouts` guard (which checks the *training* rollout
window, not the *generation* behavior). With `max_new_tokens=512` and a
reward that does not penalize short outputs, the shortest-path
policy is to emit nothing. The first attempt (`max_new_tokens=256`) was
killed by the guard at step 5 (215.7/256 = 84% of cap); the 512 cap
passed the guard but the model found the same equilibrium faster.

This is not a bug in the training loop — it is the recipe's reward
shaping. The `grpo-gsm8k-27b` recipe was designed for a different
question (does the loop survive 100 steps) and answered it. The
collapse is the answer to a different question (does the recipe train
a useful model), and the answer is no.

## Rule

A clean-arm run that survives is a survival certificate, not a quality
certificate. Any experiment that uses a clean-arm run as a control must
record the collapse: reward 0.83→0.19, tokens 105→5, by step 87 the
model emits silence. The control is a collapsed model; the treatment's
delta is measured against that.

## Results

| date | run | machine | what | result |
|---|---|---|---|---|
| 2026-09-09 | 8ae3e5ff6a95 | H20 card 1 | clean-arm 100-step, t512 | survived; reward 0.83→0.19, tokens 105→5 |
