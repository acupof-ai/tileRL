# `--prompts-per-step`: it lands and runs, and the tie it was built for is already gone — 2026-09-08

> Status: final for the shapes and the values (both arms ran on `9bb688b`, H20 card 1); the
> 1.21x pricing is **not** confirmed — see "The values run, and what it does not support".

## Context

The measured tie fraction is the constraint on P1, not accuracy. On MATH level 5, 65 of 100
problems produce a group where every sample agrees, and 59 of those 65 sit at 8/8
(the pass@k entry, `2026-09-08-the-tie-is-at-the-ceiling`, in flight on another branch --
named rather than linked so this entry does not depend on which lands first). A tied group has zero advantage and
contributes no gradient, so those steps buy nothing. The 2026-09-05 GSM8K run measured the
same thing at 0.81 with 73 of 81 tied steps at the ceiling — "81% of the run's compute bought
nothing" (`wins/2026-09-05-p1-grpo-27b-run.md`).

Every other lever on the table multiplies `seconds_per_step`. This one multiplies
`steps_to_score`, and nothing had acted on that factor.

## What Worked

**A step becomes several prompts, each still normalised within its own group.**
`group = prompts_per_step × completions_per_prompt`; the advantage is computed per prompt, and
a step is gradient-free only when **every** group ties.

**The first framing of the mechanism was wrong and worth recording.** It was "8 distinct
problems tie with probability ∏pᵢ rather than one problem's p⁸" — which needs one sample per
prompt, and GRPO cannot do that: its baseline is the group mean, so a one-sample group has no
defined advantage. The real direction is the opposite. Fewer completions makes **each group tie
more often**; the entire gain comes from a step holding several groups when only one has to be
untied. Same conclusion, inverted mechanism.

### Priced from the measured distribution, at fixed compute

**Every number in this section is `tied@λ=0` — the tie in *correctness*, with no length term in
the reward.** The shipped default is `--length-penalty 0.1`, under which these figures do not
apply; the section after this one is why. The condition is in the name because it is a
command-line default, and a reader will not think to check it.

The 8 samples of a problem are exchangeable, so `q_comp` — the chance one group of `comp` ties
— is measured by drawing `comp` of the real samples without replacement, not by fitting a
per-problem difficulty:

| split (8 rows either way) | per-group tie | step gradient-free | steps with gradient | vs 1×8 |
|---|---:|---:|---:|---:|
| 1 × 8 (today) | 0.6496 | 0.6496 | **35.0%** | 1.000x |
| **2 × 4** | 0.7621 | 0.5807 | **41.9%** | **1.197x** |
| 4 × 2 | 0.8715 | 0.5769 | 42.3% | 1.208x |

**2×4 captures essentially all of it.** The control is the `comp = 8` row: measured 0.6496
against the run's own 0.650 tied fraction, agreeing to 0.0004 — the sub-group draw has to
reproduce a number it could have contradicted.

**On GSM8K the same lever prices higher, because the tie fraction is higher.** Fitting a Beta
to that run's two published constraints — `q_8 = 0.81`, and 73 of 81 tied steps at the ceiling —
gives Beta(0.3122, 0.0616), which predicts `q_4 = 0.8631` and `q_2 = 0.9251`. At fixed 8 rows:
1×8 19.0% → 2×4 **25.5%, 1.34x**. `q_8` reproduces 0.8100, which is what makes the fit
readable. 1.21x was the figure meant to enter the decision, since P1 runs on level 5; 1.34x is
the GSM8K equivalent. **Neither survived contact with the card: the measured per-group tie moved
the opposite way (`q_4` 0.800 vs `q_8` 0.900, paired), and at the shipped λ both numerators are
zero.** Kept as the derivation that motivated the change, not as a live prediction.

**A same-sign bias does not cancel in a ratio.** The first pricing used
`E[p^comp + (1-p)^comp]` with `p_i = correct_i/8` and reported 1.37x, on the argument that all
three terms were biased high so the *ratio* survived. The bias is **−0.073 / −0.033 / −0.016**,
shrinking monotonically in `comp`, so it inflates the numerator most: 1.37x against the measured
1.21x. Consistent sign is an argument about direction; a ratio needs one about magnitude.

`q_comp ^ prompts` was checked rather than assumed — `E[X]^g ≠ E[X^g]` in general and these
problems are heterogeneous. An empirical resample gives 0.6337 against the formula's 0.6328; it
holds because prompts are drawn independently, so heterogeneity moves the variance and not the
expectation.

## The shipped default already removes the dead steps, and that is not the good news

`--length-penalty` defaults to 0.1, and it landed today (#293, `aee90e5`) — **after** the
2026-09-05 run whose 0.81 tie fraction motivated this whole change. The two do not compose the
way the section above assumes.

Measured per rollout on the card run (`/work/runs-ppsn/972b8374cf22/rollouts.jsonl`, 10 groups
× 8 rows, correctness recovered from each row's own logged reward):

| | λ = 0 | λ = 0.1 (shipped) |
|---|---:|---:|
| groups dead (every advantage 0) | **9 of 10 (90%)** | **0 of 10** |

Same rollouts, same task. In an all-correct group `λ` cancels — `adv = −(Lᵢ − L̄)/std(L)`, no
λ and no cap in it — so any spread in length unties the group, and no group here had 8
completions of one length. **So the premise "81% of steps are gradient-free" is false under the
shipped default.**

**0 of 10 is an observation, not an identity.** Eight all-correct completions of identical
length would still tie; step 9's spread was 7 tokens. A high-probability observation and an
identity get handled differently — the latter would let the check be deleted.

**And the dead steps were not replaced by useful ones.** The gradient those 9 steps now carry
is a *length* gradient: it teaches the model to shorten answers that are already right. The
2026-09-05 run moved tokens/correct 394.0 → 143.8 (2.74x) while accuracy moved 88.0 → 93.6
(+6%) — and that was at λ=0, so λ=0.1 only tilts it further. The waste did not go away; it
changed from *no gradient* to *gradient pointing at length*.

**Worse, that length gradient does not weaken as the length spread shrinks.** The advantage is
a z-score, so its scale is `std(L)` — the group's own spread — which divides out:

| step | tok spread | adv std | matches −(L−L̄)/std(L) |
|---:|---:|---:|---|
| 2 | 625 | 1.000 | yes, to 1e-6 |
| 9 | **7** | **1.000** | yes, to 1e-6 |
| 1 | 791 | 1.000 | no — the one mixed group, correctness still in the reward |

A group whose 8 right answers differ by 7 tokens gets the same full-scale gradient as one
differing by 625. **Normalisation removes the information about whether the length difference
meant anything.** `_length_aware`'s docstring states that λ cancels in an all-correct group, so
that part is known and deliberate; what is recorded here is the consequence, which is not in
it. Not proposing a change to the reward — that is a design decision, and it needs the GSM8K
curve first.

**So the lever's claim has to be restated, and it gets weaker.** Not "recover the 81% of steps
that are gradient-free" — at λ=0.1 there are almost none. The open question is whether more
groups per step move the gradient off the length axis and back onto the correctness axis, and
**answering it needs the gradient decomposed by axis, not a tie fraction.** 1.21x does not
apply to it: both its numerator and denominator are dead-step fractions.

None of the three raises, and none changes a shape. Each was reverted and the suite re-run.

**1. Normalising across prompts.** `reshape(-1, group)` infers the group count from the length,
so 16 rewards at `group=8` become `(2, 8)` whether the caller meant two prompts or one prompt
with twice the rollouts. Both are legal shapes now, so **the length alone cannot distinguish
them** — only the caller's own expectation can, which is why `groups` is a parameter rather than
a derived value. On the test fixture the correct reading is `[+1, −1, +1, −1]` and the buggy one
is `[−0.896, −1.095, +1.095, +0.896]`: both of the easier prompt's rows go positive, i.e.
gradient for being the easy prompt. Two rows differ in **sign**, not magnitude.

**2. The seed stride.** The one-prompt loop strided by `group`; a step now consumes
`prompts_per_step × group` seeds, so striding by `group` reissues the previous step's seeds to
every prompt after the first — identical completions, silently.

**3. The prompt-length mask.** `plens` was `np.full(group, len(prompt))`, a scalar. With prompts
of 3 and 6 tokens it masks the second prompt's rows against the first's length, so three prompt
positions enter the scored span and three completion positions leave it. The gradient changes;
the shape does not.

**The assertion for (2) was itself vacuous on the first attempt.** It recomputed the seed from
train.py's formula and compared — a test that mirrors the expression it guards passes whatever
that expression becomes, and it was **green on the mutant**. Rewritten to read the seeds the
engine was actually handed (`params.seed` off a wrapped `submit`), it goes red. Same for (3):
asserted from what `rl_step` received, not from the fixture's own prompt lengths.

## Also landed: `tied` per curve point

`eval_curve` now records the run's tie fraction over the steps since the previous point. The
2026-09-05 run went **0.50 → 0.50 → 0.87** across its own steps while mean reward went
**0.675 → 0.825 → 0.975**, so "the groups stopped disagreeing" and "the policy learned" are
confounded in any single aggregate. Per point they separate: `score` flat with `tied` rising is
the usable set self-consuming; both flat is another cause.

**This is a diagnostic, not evidence for the exhaustion story.** The pass@k measurement it came
from is a t=0 cross-section on the base policy and cannot test a trajectory claim at all.

## Rule

**A quantity whose meaning depends on a flag needs that flag in its name.** `tied` counts
correctness ties at `λ=0` and is ≈0 regardless of task difficulty at `λ=0.1`. Same field, same
name, two different quantities — and λ is a *default*, so nobody reading a tie fraction thinks
to check it. Write `tied@λ=0`.

**A lever priced against a configuration that has since changed is priced against nothing.**
1.21x assumed 65% of steps carry no gradient. The default that landed the same day makes it 0%.
The pricing was correct when computed and was never re-derived against the tree it would ship
into.

**Removing the symptom is not removing the waste.** The length term removed the dead steps by
giving all-correct groups a length gradient. The steps are no longer free, but what they buy is
shorter right answers, and because the advantage is a z-score its strength does not fall as the
length spread shrinks — 7 tokens of spread produces the same full-scale gradient as 625.

**Two arms that consume different numbers of prompts are not two arms.** At
`prompts_per_step=2`, ten steps draw twenty problems; at 1, ten. The unpaired per-group tie read
0.650 vs 0.900 — the direction opposite the model — and paired on the ten shared problems it is
0.800 vs 0.900 with one discordant pair. **Most of that gap was the extra problems.** The general
form: **align on the quantity the measurement consumes, not the one you happen to be counting.**
Here the steps were equal and the problems were not.

**When two shapes are both legal, the length cannot tell them apart.** `(2, 8)` was a bug
yesterday and is a valid step today, so the guard has to compare against what the caller
intended, not against what fits.

**A test that recomputes the expression it guards cannot fail.** Read the value the system
actually used — the seed the engine was handed, the mask `rl_step` received — not the formula
that produced it.

**An exit code cannot carry a four-way outcome.** rc=1 held both a real death and a gate that
is vacuously false at `steps=1` (`reward_rises`, first == last); rc=0 held both a real pass and
five gradient-free steps. Read the fields.

## Results

| date | commit | machine | target | model | change | 1×8 | 2×4 | note |
|---|---|---|---|---|---|---|---|---|
| 2026-09-08 | pending | — | cpu | tiny | correctness + 3 mutants | pass | pass | 513 passed, ruff clean |
| 2026-09-08 | 1a49186 | H20 card 1 | cuda | Qwen3.8-27B NVFP4 | **shapes only** | rc=0 | rc=0 | 60.93 / 53.07 s/step, peak 45.38 / 45.33 GiB |
| 2026-09-08 | 9bb688b | H20 card 1 | cuda | Qwen3.8-27B NVFP4 | **values, 10 steps** | see below | see below | 29.70 / 35.70 s/step, peak 55.78 / 55.75 GiB |

### The values run, and what it does not support

GSM8K, `--max-new-tokens 1024`, 10 steps per arm, one revision, one sitting. `tok` ran 51–457
against the 1024 cap, so the cap is not binding — the earlier attempt at 512 truncated
everything (`tok 512`, `tied 1.00`, every advantage zero) and measured nothing.

The three changed paths executed with a nonzero advantage on every step of both arms, so the
values half of the card claim is now discharged. Per-group ties recovered from each row's own
reward, and per-*step* gradient-freedom (a step is dead only when **every** group in it ties):

| | groups tied @λ=0 | steps gradient-free @λ=0 | steps gradient-free as run (λ=0.1) | median s/step |
|---|---:|---:|---:|---:|
| 1×8 | 9/10 = 0.900 | 9/10 | **0/10** | 29.70 |
| 2×4 | 13/20 = 0.650 | 4/10 | **0/10** | 35.70 |

**The step-level mechanism reproduces exactly.** `q_4² × 10 = 4.2` steps predicted dead, 4
measured — the claim that a step ties only when all its groups tie is the one thing here that
was tested and held.

**The per-group direction came out opposite the model, and the comparison is confounded.** The
model needs `q_4 > q_8` (0.863 vs 0.810); measured unpaired it is 0.650 vs 0.900. But the two
arms did not see the same problems: at `prompts_per_step=2` a 10-step run consumes 20 prompts,
at 1 it consumes 10. **Paired on the 10 prompts both arms actually shared, `q_8 = 0.900` and
`q_4 = 0.800` with one discordant pair of ten — McNemar p = 1.0, no detectable difference.** So
the unpaired 0.650 is mostly the 10 extra problems arm B drew, not the group size.

**Nothing here confirms or refutes 1.21x.** 10 steps gives 10 (or 20) groups; the dead-step
contrast 9/10 vs 4/10 is Fisher p = 0.057, and the paired group contrast has one informative
pair. A 5x figure is computable from these numbers (0.60/35.70 against 0.10/29.70) and it is
**not reported as a result**: it divides two rates each measured on ten steps, on different
problem sets, on the axis the shipped default has already flattened.

**And it is the wrong axis anyway.** Both arms ran at λ=0.1, where 0 of 10 steps were
gradient-free in either. Everything in the table's λ=0 columns is a *counterfactual* recomputed
from the same rollouts — useful for checking the mechanism, not a measurement of the shipped
configuration. What would settle the lever is a decomposition of the gradient by axis, which
this run does not produce.

**Arm B costs 1.20x per step** (35.70 vs 29.70 s median). The earlier cap-512 pair had the
ordering reversed (53.07 vs 60.93), so per-step cost is not stable across configurations at
n=10 either.

### The shapes-only run, kept because it is separate information

`prompts_per_step` 1 and 2 both run on 27B + NVFP4 + `decode_graph=True`; the engine sizes
correctly and the two splits reach different branches at the same peak (45.38 vs 45.33 GiB,
5 steps each, rc=0). Its **values half measured nothing**: `--max-new-tokens 512` against a
1029-token mean truncated every rollout, so every reward was −0.1, `tied_group_fraction` was
1.00 in both arms, and every advantage was zero. It returned **rc=0** on five gradient-free
steps, which is why rc is not the signal and the verdict script reads `tied`, `reward` and
`tok` instead.

*The verdict script, and why it distinguishes three cases:* a run is VOID unless at least one
step per arm has `tied < 1.0`, and a VOID run must name which tie it hit, because the remedies
differ — `tied 1.00` at the floor with `tok` at the cap is truncation (raise the cap), at the
floor with `tok` well under the cap is too hard for the policy (easier task), at the ceiling is
saturation (harder task). Only the first is an instrument fault; the criterion that separates
them is *would changing an instrument setting make it go away*, not *is it a bug*. The `tok`
test is `0.98 × cap`, not `== cap`: `tok` is the group mean, so one row emitting EOS a token
early puts a fully truncated group under the cap.

**Even a passing values run only shows the paths executed with effect and did not crash.** A
wrong seed stride or a wrong mask trains worse without going red on a card. Correctness rests
on the three CPU mutants above.

*Two failures before the first values step, each costing a full attempt:* the bare hub id
`Qwen/Qwen3-27B` cannot resolve on this pod (no network — the weights are at
`/work/Qwen3.8-27B-NVFP4`), and the fp4 GEMM does not codegen under the pod's default
`python3` (tilelang 0.1.8): 8 × `no instance of overloaded function "tl::tma_load"`. Both are
environment, not this change; the working peer runs put `/work/tl013/bin` (0.1.13) on `PATH`
and set `TILERL_QWEN38_SOURCE`. A one-step smoke test costs 4 minutes and would have caught
both.

The card arm needs the training engine sized for `prompts_per_step × group` rows. That sizing's
length axis was a separate open defect when this was written and landed meanwhile (#322, per
consumer rather than per axis), so this branch rebased onto it makes the row axis here the only
change. Verified across the three 8-row splits at `--eval-max-new-tokens 2048`: 1×8, 2×4 and
4×2 all size to **1328 blocks**, which is the point — the same rows at the same lengths must
cost the same pool, so the comparison is not confounded by memory.

Both arms ran in one sitting on one revision (`9bb688b`). Measuring 1×8 now and 2×4 after the
intervening changes would have traded a cap confound for a version confound. **What it did not
control is the problem set** — a 10-step run at `prompts_per_step=2` consumes 20 prompts and at
1 consumes 10, so the arms are only comparable on the 10 they share. That is fixable: draw both
arms from the same prompt list of the same length, i.e. give the 1×8 arm twice the steps.

## Open, and what would answer it

**The tie-fraction axis is settled enough not to need another run on it.** The mechanism
(a step dies only when all its groups die) reproduced to 4.2 vs 4; the per-group direction is
unresolved at this n and is the wrong axis under the shipped default anyway.

**What is open: at λ=0.1, do more groups per step move the gradient off the length axis and
onto the correctness axis?** That needs the gradient decomposed by axis — the correctness term
and the length term measured separately — not a tie count. Not started.

**And it may be moot.** A peer's GSM8K curve reached 93.2% at step 25 against the 100-step
anchor's 93.6%, at λ=0. If the run tops out at a quarter of its steps, the steps this lever
recovers can be skipped outright, and early stopping (~4x) competes with splitting (1.21x) for
the same wall clock. That comparison is only meaningful inside one curve, so it waits for that
run's step 100 — a cross-run ratio would not settle it.
