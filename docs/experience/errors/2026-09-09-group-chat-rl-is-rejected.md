# Group-chat RL is rejected for the roadmap — 2026-09-09

**Status:** rejected. Survey of `cklxx/Meshy` (= `OpenBMB/Meshy`, zero divergent
commits) and the group-chat RL literature, at ckl's request after forking Meshy.

## The core distinction

GRPO's group is **N independent rollouts of one prompt** — causally unrelated
samples, each a control for the others. A group chat's group is **N agents in one
conversation** — a causal chain where agent i's output is conditioned on agents
1..i−1. Porting `group_advantages` across the members of one chat subtracts a
baseline from causally dependent quantities; the result has no interpretation as
a policy-gradient estimate.

## Three reasons to reject

1. **It does not touch `time_to_score`.** An episode is N agents × multiple turns
   — strictly more generated tokens than one GSM8K rollout, so `seconds_per_step`
   rises. `steps_to_score` is not demonstrably shortened: the score is still
   single-agent GSM8K accuracy, and chat RL's reward signal is diluted across
   agents. Both factors move the wrong way or are unmeasured.

2. **Meshy has no multi-agent machinery.** Read in the code, not the README: the
   only "multi-turn" is tokenization (chat templates put EOS after every turn of a
   single rollout's prompt history). `group_size` rollouts per prompt, same as any
   GRPO engine. Its advantage functions do `reward − mean` over a prompt's rollout
   group — the same seam as ours. It is a service-per-layer async RL engine whose
   infrastructure was already rejected at a measured 1.36x bound
   ([async-rl-ceiling](2026-09-08-async-rl-ceiling-and-what-the-reference-runs.md)).

3. **The tied-group failure mode gets structurally worse.** GSM8K already sits at
   `tied_group_fraction` 0.81 (73/81 steps tied at the ceiling). For a Bernoulli
   episode outcome the all-success tied fraction is monotone in the success rate —
   if group chat delivers its only motivation (higher success rate), it makes tied
   groups *more* common, not less. The N agents in one episode are correlated by
   construction, so the baseline's effective sample count is K (episodes), never
   N×K (agent-positions).

## Adopt criteria for the future

All three required, in order:

1. A chat task is in the objective, set by ckl — not inferred from a fork.
2. A per-agent reward stream (the judge path) or a repeated-episode design with
   K ≥ 4 and an outcome that is not tied. Joint-reward self-play without this is
   longer-prompt GRPO and should be evaluated as that, not as a new method.
3. A measurement that the chat system's success rate beats the single agent's on
   the same task.

When adopted, the minimal path needs no engine change: self-play on one engine, K
repeated episodes as the group, per-position advantage across episodes, `judge.py`
for within-episode differentiation, per-turn loss masking on the tape.

## Rule

A "group" in GRPO is a statistical device (independent samples for baseline
subtraction). A "group" in a chat is a causal chain. The same word does not make
the same method — and a fork is not a roadmap.
