# Scope: the 2.74x tokens/correct is a cap-256 training result, and no gate can see the cap — 2026-09-08

**Date:** 2026-09-08
**Machine:** local, reading the tree. No card, no new run.
**Subject:** the scope of [P1's 2.74x](2026-09-05-p1-grpo-27b-run.md), and what six gates
cannot detect about the eval cap.

## Context

P1 measured GSM8K +5.6 points and tokens/correct 394.0 → 143.8 (2.74x), and the entry says
the length effect is "the larger effect and the one this run actually bought". That is the
number people will carry forward. It needs its scope attached, because the curve that
produced it is a curve of a specific training configuration, not of GRPO on GSM8K.

## The curve measures cap-256 training

`grpo-gsm8k-27b` trains at `max_new_tokens=256` (recipes.py:20) and evaluates at
`eval_max_new_tokens=2048` (:22). The base policy's own completions on that eval average
346.7 tokens (173352 / 500). **Rollouts are cut at 74% of the length the policy wants**, so
within every group the short completions are the ones that finish, and the reward — pure
`answer_match`, no length term at the time — pays only the finishers. That is brevity
pressure produced by the cap, not by the objective.

At cap 2048 the pressure is absent by construction: the policy's 346.7-token mean sits at
17% of the cap, almost nothing truncates, and length stops discriminating inside a group.
So the 2.74x does not transfer to a 2048-cap run, and neither does the reading that "the
policy learned to be short" — it learned to fit under 256.

**The tree now refuses to reproduce it.** `_refuse_short_rollouts` (cli.py:524) landed at
`4cf169b`; P1 launched from `91977a8`, which is its ancestor. Run the recipe today and it
exits before step 1:

```
error: the base policy averages 347 completion tokens but --max-new-tokens is 256.
Raise the cap above 433, pick an easier task, or pass --allow-short-rollouts ...
```

Driven through the real guard, not restated from its source. The guard refuses at both eval
caps — 346.7 and the artifact's 238.7 are each above the 204.8 threshold — so the artifact
below does not hide the guard, and the recipe's cap is unrunnable on its own terms either
way.

## The cap-256 eval artifact passes every gate

Score the same run at `--eval-max-new-tokens 256` and the base arm reads 38.4% instead of
88.0%, because the base policy averages 238.7 tokens against a 256 ceiling and is cut
mid-derivation. The delta becomes +53.4 points instead of +5.6 — **9.5x inflated, direction
correct, and it looks like a triumph.**

Six gates run on an RL manifest. None of them goes red:

| gate | reads | why the cap is invisible |
|---|---|---|
| `gsm8k_improves` | `gsm8k_after` vs `before + 0.05·n` (cli.py:1074) | **both arms use the same cap.** 459 ≥ 217 passes, exactly as 468 ≥ 465 does |
| `mmlu_holds` | `mmlu_after` | MMLU scores one token — `max_new_tokens=1`, hard-coded at eval.py:85. No cap reaches it |
| `reward_rises` | training reward windows | training cap, not eval cap |
| `groups_untied` | `tied_group_fraction` | rollouts, not eval |
| `rollouts_within_cap` | 5-step rollout mean | rollouts, not eval |
| `ce_falls` | `ce_first` | never written by the GRPO branch; reports *unmeasured*, so it cannot go red at any cap |

**The margin is the part that makes it worse than a silent pass.** At cap 2048 the run
clears `gsm8k_improves` by 3 questions of 500. At cap 256 it clears by 242 — **81x the
margin**. The broken configuration does not scrape through looking marginal; it produces
the most convincing-looking result in the file.

This is what "cannot be discovered from the curve itself" means precisely. Every gate is
a comparison between two arms sharing the misconfiguration, so the misconfiguration
cancels. Only a second measurement at a different cap exposes it, and nothing asks for one.

Credit: 55 stated the all-six-gates claim; the table above is my check of it, gate by gate
against the source.

## The manifest wording

Written into `recipes.py` beside the flag, because a config that records the choice and not
the reason gets reverted by the next person saving 66 minutes:

> `--eval-max-new-tokens 2048` is not a tunable. At cap 256 the base arm reads 38.4%
> instead of 88.0% — 49.6 of those points are the cap truncating the base, not a model
> difference. The whole curve comes out correct in direction, ~9x inflated, and extremely
> convincing. The 66 minutes saved buys an artifact the curve itself cannot reveal.

## The column that discriminates, and why it is not stored

`tokens/correct` separates the two things a rising score can mean: **score up and
tokens/correct flat = the policy is learning to be right; both moving = it is mostly
learning to be short.** P1 is the second case — +5.6 points (inside a ±4.5 interval) against
2.74x on length.

It is computed, not stored: `mean_len × total / correct` from three fields the manifest
already has. 55's reasoning, which is right — a fourth stored field can contradict
`mean_len`, `correct` and `total` while all four sit in legal ranges individually, and no
check would catch it. `gsm8k_accuracy` already returns `ntok` (eval.py:154) and cli.py:715
already logs the ratio.

## Two mistakes of mine, which are one mistake

**Two individually-correct numbers placed side by side is an unchecked assertion that they
share a population.** I used 229.2 s/step against a GSM8K schedule. 229.2 is real — run 2,
`grpo-math-27b`, MATH level 5 at a 2048 cap. The GSM8K figure is 56.88. Neither number was
wrong; the pairing was, and nothing in either number's presentation says which population it
came from. Same shape as a units error, with the dataset as the mismatched dimension. The
2.74x above is the same hazard one step later: correct, and meaningless once it is read
next to a 2048-cap run.

**"Not in the tree" and "not in the two files I read" are different claims.** I said the
eval wall clock was not measured. It is: `eval_before_secs` / `eval_after_secs`, written at
cli.py:691 and :732. I had read two entries, found nothing, and reported the absence at
global scope.

Both are the same move — a local observation promoted to a general one without the step
that would license it. The first treats two sources as one population; the second treats one
search as the whole tree. Neither announces itself, because in both cases every component
statement is true.

## The truncation is measured, not a risk

Added 2026-09-08 evening, from the live curve run (55, `12da5a0`), which trains at the same
cap 256:

- **GSM8K rollouts at cap 256 average 322 tokens.** The cap sits below the mean, not above
  it.
- **3 of the first 17 steps had all eight rollouts at 256** — reward 0.0, tied 1.00, floor
  ties. **18% of steps.**

So the entry's "brevity pressure comes from the cap" is not a mechanism argument any more;
it is the observed rate at which the cap wins outright. A group that ties at the floor
contributes no gradient for the same reason a group that ties at the ceiling does not, and
the two are indistinguishable in `tied_group_fraction` alone — 10 of those 17 steps read
`tied 1.00`, 7 at the ceiling (reward 1.0, saturated) and 3 at the floor (reward 0.0,
truncated). Telling them apart needs `at_cap`, which #323 added; before it, "the curve went
flat" had two readings with opposite remedies.

**The second curve point reproduces the 2.74x.** At step 25: 346.5 → 117.8 tokens, **2.94x**,
with 0 of 500 eval completions at the 2048 cap. Same training cap, same length collapse,
independent run. That is confirmation of the scope claim, not a counterexample to it —
both runs train at 256.

## Rule

**A measurement's scope includes the configuration that produced it, and a training cap is
part of that configuration.** The 2.74x is a fact about training at 256 against a policy
that wants 347. Quote it with the cap or do not quote it.

**A gate comparing two arms cannot see a fault both arms share.** Six gates, one shared eval
cap, zero coverage — and the fault makes the passing margin 81x wider, so it reads as
success rather than as noise. Detecting this class needs a second measurement at a different
setting, which no gate structured as before-vs-after will ever ask for.

**Before putting two numbers in one sentence, name the population each came from.** If the
answer differs, the sentence is an assertion, not an observation.

**Report an absence at the scope you searched.** "Not in the two files I read" is a finding.
"Not in the tree" is a different claim that needs the tree.
