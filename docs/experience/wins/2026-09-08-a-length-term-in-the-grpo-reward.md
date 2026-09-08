# A length term in the GRPO reward: the tie is broken, the collapse is unproven — cpu, 2026-09-08

**Date:** 2026-09-08
**Machine:** local CPU target, tiny model. **No card** — the V100 has held pid 3171701
(25978 MiB) all session and its ownership is unresolved.
**Change:** `src/tilerl/cli.py` — `_length_aware` wraps the RL reward closure; `--length-penalty`
(default 0.1).

> **Two claims, two scopes, and the second one is not made.** CPU and tiny prove that an
> all-right group stops tying and its advantages order by length. They **cannot** prove the
> policy stops lengthening — that needs a real run, `per_rollout`'s 8 pairs per step, and the
> within-group correlation shrinking over training. That half is `pending-remote`, named below,
> and the green CPU result does not stand in for it. #131 wrote up a guard as making a failure
> "impossible by construction" and run 2 failed by another route; this entry claims what it
> measured at the scope it measured it.

## Context

[Run 2](../errors/2026-09-06-the-rollouts-grew-into-the-cap.md) died at step 45 of 100: the
policy collapsed onto the 2048-token rollout cap at step 41 and stopped producing a gradient.
Its three fixes — padding buckets (#140), a periodic drift guard (#143), and "a length term in
the reward, or a length-aware advantage" — left the third open, and the entry calls it "the
actual cause"; the other two only contain it.

Three prerequisites were already in the tree and were read rather than rebuilt:
`grpo_loop(per_rollout=)` logs 8 `(length, reward)` pairs per step, `_within_group_r` pools
within-group deviations so prompt difficulty cannot confound the correlation, and
[the wins entry](2026-09-06-what-a-length-term-can-recover.md) had already bounded the fix.

## What Worked

**The defect, on demand, before the fix existed.** Two correct completions 77.7x apart in
length:

| assertion | today's code |
|---|---|
| reward of each | `True` / `True` — indifferent |
| all-right group of 8 → advantages | all `0.0`, tied, no gradient |
| **negative control:** mixed group `[1,1,1,1,0,0,0,0]` | `[+1,+1,+1,+1,−1,−1,−1,−1]` |

The negative control is what makes the zeros mean *indifference* rather than a broken
`group_advantages`.

**The term goes in the RL reward closure and not in `MATCHERS`.** `cli.py` passes
`MATCHERS[args.reward]` into `gsm8k_accuracy`, and that count becomes
`manifest["metrics"]["gsm8k_*"]` — the number P1's exit criterion reads ("GSM8K held-out 500 q,
after − before ≥ +5 pt"). A length term in the matcher contract would sit inside the north-star
gate. `_length_aware` is the seam that is the reward and not the metric.

**The synthetic reward is deliberately left alone**, and not because it is the smoke-test path:
`sum(1 for t in completion if t < half) / max(len(completion), 1)` is a **rate**, so its
expectation does not grow with length and the defect is absent by construction. A longer
completion earns no more, so nothing pressures the policy to lengthen. That holds even if the
path is later promoted.

**λ is inert exactly where the defect mostly lives.** In an all-right group it cancels: with
`r_i = 1 − λL_i`, `r_i − mean = −λ(L_i − L̄)` and `std = λ·std(L)`, so
`adv = −(L_i − L̄)/std(L)` with no λ. Driven through the real `group_advantages` at
λ ∈ {1e-4, 1e-2, 0.1, 0.5, 1.0} on lengths `[412, 1893, 655, 1204, 988, 1560, 301, 2048]`:

```
every λ:   +1.164465 -1.228697 +0.771799 -0.115336 +0.233701 -0.690599 +1.343831 -1.479163
max|Δ| across four orders of magnitude:  1.6e-12
same against the closed form -(L-L̄)/std(L): 1.6e-12
```

**But λ is a real dial in mixed groups**, so it is bounded rather than free: same lengths with
correctness `[1,1,1,1,0,0,0,0]` moves `max|Δ|` 9.1e-02 from λ=1e-4 to 0.1, and 7.6e-01 to
λ=1.0. The bound is the ordering it must preserve — inversion needs `λ > cap/gap` and the worst
gap is `cap − 1`:

| λ | correct-at-cap | best wrong (length 1) | inverted? |
|---|---|---|---|
| 1.0 | +0.00000000 | −0.00048828 | no |
| 1.0004885 | −0.00048850 | −0.00048852 | no |
| 1.001 | −0.00100000 | −0.00048877 | **yes** |

The threshold lands between 1.0004885 and 1.001, against the closed form
`2048/2047 = 1.000488520`. **So λ ∈ (0, 1.0] is safe throughout and nobody should sweep it** —
a tie-fraction sweep would show nothing on 14 of the 19 target steps and would be measuring a
quantity that provably does not vary there. The flag exists to set λ = 0, not to tune.

## The ceiling, stated before anyone asks

**19 of run 2's 45 steps produced no gradient (42.2%, 1.08 h of 2.71 h). This reaches at most
17.** Steps 41 and 44 had the group mean exactly at the cap, so all 8 rollouts are truncated to
the same width and *any* function of length is constant within the group. Those two are
recoverable only by raising the cap, which is fix 4 and comes after all three. A fix advertised
at 19 that reaches 17 gets re-litigated.

## The gate

`test_a_length_term_breaks_an_all_right_group_and_lambda_cancels_there` — the failure case, its
negative control, the shortest-first ordering, λ invariance to 1e-9 against the closed form, λ's
effect in mixed groups, the inversion boundary, and the at-cap case staying zero.

`test_the_rl_reward_closure_carries_the_length_term_and_the_matcher_does_not` — drives the real
`_length_aware` rather than a copy of its arithmetic, and asserts `λ=0` reproduces the old
behaviour exactly.

**Mutation-checked, two mutants for the two ways this can be wrong:**

| mutant | result |
|---|---|
| drop the length term (`return r`) | closure test **red** at the long-vs-short assertion |
| move the term **into** `MATCHERS` (option B) | closure test **red** at `λ=0` → 0.984 ≠ 1.0 |

The second is the one worth having: it is the design that was rejected, and it is red because
the test asserts the matcher is untouched, not merely that the reward has a length term.

24 passed in `tests/test_rl.py` (22 before), `ruff check` clean.

## Results — no bench run, and the argument for why

The change is one subtraction and one division per rollout, off the forward path: 8 float
operations per step against run 2's **216.8 s mean step**. `advantages` keeps its shape and
`rl_step` sees the same tensor contract, so no kernel, no launch and no allocation changes. That
is the argument; it is not a measurement, and it is stated as an argument.

**Gradcheck does not apply**, said explicitly rather than left unmentioned: the change produces
reward scalars only, adds no backward path to the tape, and leaves the `advantages` shape
unchanged.

## `pending-remote` — the half that is not measured

**Claim not made:** that the policy stops lengthening. **What would settle it:** one real run
with `per_rollout` logging, and `_within_group_r` over its rows showing a negative within-group
correlation that shrinks over training. Two group means per step cannot express it — that is
why run 2's own mechanism sentence is on the wrong axis and no re-analysis of that run can move
it onto the right one.

**Also unmeasured:** whether λ = 0.1 is a good default in the only place it matters (mixed
groups). The bound says every value in (0, 1.0] is order-preserving; it does not say which
converges fastest. That is a question for a run, not a CPU sweep.

## Rule

**Bound a reward term by what it can reach before writing it, and put the ceiling in the PR.**
The analysis that said 17 of 19 was already in the tree; reading it removed a day of work and
supplied the honest number.

**When a coefficient's effect is normalised away, that is a fact about the coefficient, not a
caveat.** λ cancels in all-right groups exactly. The instinct to tune it is measuring a constant
in 14 of 19 target steps, and the algebra says so before any sweep does.

**A mutant for the design you rejected is worth more than a mutant for the bug.** Removing the
term tests that the feature exists. Moving it into the matcher tests that it is in the *right
place* — and the wrong place is the one that would have put a length term inside P1's gate.
