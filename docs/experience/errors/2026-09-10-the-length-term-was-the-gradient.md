# The length term was the gradient from step 1

2026-09-10. `grpo-gsm8k-27b` P1 measurement, run `b4a4e7b23ab8` at `ca648a71`, card 1.

## Context

P1's gate was +5 pt GSM8K over the 91.4% base — judged unsatisfiable before this
run (96.4% means cutting the error rate 58%), so the run was reclassified as a
measurement: does GRPO move GSM8K at this base at all? Recipe: 100 steps, group 8,
LoRA r16, lr 1e-4, seed 0. The rollout-length guard refused the recipe's 256 cap
(the base policy averages 347 completion tokens on the train set); the run used
512, the guard's own prescription.

before 458/500 = 91.6% (eval-cache hit — same sha, eval set and params as the
row the same morning) → after 391/500 = 78.2%. **Δ = −13.4 pp, se 2.2, z = −6.09,
McNemar b=94, c=27.** Curve points (n=100, health checks only): 95/63/78/83 at
steps 25/50/75/100. Rollout mean length: 329 tokens at step 1 → 14 at step 100.

## Root Cause

The reward is `correctness − 0.1 · len/512`. Decomposing the advantage by
component — `corr = reward + 0.1·tokens/512` reconstructs exactly from
`rollouts.jsonl`, then the shipped `group_advantages` runs on each component
separately:

| phase            | mean \|adv_corr\| | mean \|adv_len\| |
|------------------|-------------------|------------------|
| step 1           | 0.661             | 0.857            |
| steps 2–25       | 0.384             | 0.793            |
| steps 26–100     | 0.292             | 0.631            |

The length term carries 1.3× the correctness gradient at step 1 and ~2.2× after.
What turns "shorter" into "worse" is which saturated group a rollout lands in.
When correctness is constant within a group the advantage is decided entirely by
length, and "constant" has two cases:

- **all-correct group** → the shortest *right* answer wins → learns brevity,
  accuracy untouched.
- **all-wrong group** → the shortest *wrong* answer wins → the gradient actively
  rewards giving up fast.

`tied_correctness = 0.6` says 60% of groups are one of these two, and the
all-wrong half is the engine of the collapse: 14 tokens is not brevity, it is
surrender. The model followed the dominant signal, and 14-token answers score
78.2% on GSM8K.

Two precision boundaries on the table above. `group_advantages` normalizes
within a group (subtract mean, divide std), so `adv_total = adv_corr + adv_len`
is **exact only in saturated groups** — there corr is constant, `adv_corr = 0`,
and lam cancels in the division. In mixed groups the std of a sum is not the
sum of stds, so the separately computed `mean|adv_corr|` / `mean|adv_len|` are
a **magnitude comparison**, not an exact decomposition of the gradient that
flew. The conclusion does not lean on the difference: a 1.3×/2.2× magnitude
ratio plus the exact identity in the 60% of groups that are saturated is
enough.

The reward's design assumed correctness carries the gradient and the length term
only unties saturated groups. At a 91.6% base most groups are saturated or nearly,
and even in mixed groups the length spread dominates the correctness spread. The
run measured the reward, not GRPO.

## Fix

No retry of this recipe. A P1 retry needs a reward where correctness carries the
gradient at a >90% base. `lam=0` + tiebreak is the existing lever, but lam=0 has
its own collapse history (the 2026-09-09 all-empty run), so the reward design is
ckl's call, not a relaunch. The decomposition above is computable post-hoc from
any run's `rollouts.jsonl`; for the next GRPO run on a high-baseline task it
should be a pre-registration item, not a post-mortem.

## Rule

Before trusting a GRPO run on a high-baseline task, decompose the advantage by
reward component on the recorded rollouts. If a secondary term's mean |adv|
exceeds the primary's at step 1, the reward is mis-shaped — the run measures the
reward, not the method.
