# `--prompts-per-step`: the tie is at the ceiling, and this is the lever that reaches it — 2026-09-08

> Status: pending-remote (CPU correctness done and mutation-tested; the 1×8 vs 2×4 card
> comparison waits on the training engine's KV sizing, owned by another session).

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

**On GSM8K the same lever is worth more, because the tie fraction is higher.** Fitting a Beta
to that run's two published constraints — `q_8 = 0.81`, and 73 of 81 tied steps at the ceiling —
gives Beta(0.3122, 0.0616), which predicts `q_4 = 0.8631` and `q_2 = 0.9251`. At fixed 8 rows:
1×8 19.0% → 2×4 **25.5%, 1.34x**. `q_8` reproduces 0.8100, which is what makes the fit
readable. **1.21x is the figure that enters the decision**, since P1 runs on level 5; 1.34x is
what it would be worth back on GSM8K.

**A same-sign bias does not cancel in a ratio.** The first pricing used
`E[p^comp + (1-p)^comp]` with `p_i = correct_i/8` and reported 1.37x, on the argument that all
three terms were biased high so the *ratio* survived. The bias is **−0.073 / −0.033 / −0.016**,
shrinking monotonically in `comp`, so it inflates the numerator most: 1.37x against the measured
1.21x. Consistent sign is an argument about direction; a ratio needs one about magnitude.

`q_comp ^ prompts` was checked rather than assumed — `E[X]^g ≠ E[X^g]` in general and these
problems are heterogeneous. An empirical resample gives 0.6337 against the formula's 0.6328; it
holds because prompts are drawn independently, so heterogeneity moves the variance and not the
expectation.

## Three silent failures on this path, all mutation-tested

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

**A lever on the number of steps is not comparable to a lever on the cost of a step until both
are measured on the same task.** 1.21x on level 5 and 1.34x on GSM8K are the same change; the
tie fraction is what differs, and a higher tie fraction makes this lever worth more.

**When two shapes are both legal, the length cannot tell them apart.** `(2, 8)` was a bug
yesterday and is a valid step today, so the guard has to compare against what the caller
intended, not against what fits.

**A test that recomputes the expression it guards cannot fail.** Read the value the system
actually used — the seed the engine was handed, the mask `rl_step` received — not the formula
that produced it.

## Results

| date | commit | machine | target | model | change | 1×8 | 2×4 | note |
|---|---|---|---|---|---|---|---|---|
| 2026-09-08 | pending | — | cpu | tiny | correctness + 3 mutants | pass | pass | 513 passed, ruff clean |
| 2026-09-08 | 1a49186 | H20 card 1 | cuda | Qwen3.8-27B NVFP4 | **shapes only** | rc=0 | rc=0 | 60.93 / 53.07 s/step, peak 45.38 / 45.33 GiB |
| pending-remote | — | H20 | cuda | Qwen3.8-27B NVFP4 | gradient-bearing steps | — | — | predicted 35.0% → 41.9% |

**The card row is two separate claims and they are not both established.**

*Shapes:* `prompts_per_step` 1 and 2 both run on 27B + NVFP4 + `decode_graph=True`; the
engine sizes correctly and the two splits reach different branches at the same peak
(45.38 vs 45.33 GiB, 5 steps each, rc=0).

*Values:* **not executed.** That run used `--max-new-tokens 512` against a base policy whose
mean completion on this file is 1029 tokens, so every rollout truncated, every reward was
−0.1, `tied_group_fraction` was **1.00 in both arms**, and every advantage was therefore
zero. Per-prompt normalisation, the seed stride and the per-row prompt mask all executed with
no effect on any gradient. The run proves the shapes and says nothing about the values.

*What a value run has to satisfy, and why the assertion exists:* the cap-512 run returned
rc=0 on five gradient-free steps, so rc is not the signal. A value run is **VOID unless at
least one step per arm has `tied < 1.0`**, checked by a script that exits 1. And a VOID run
has to say which of the two ties it hit, because the remedies are opposite: `tied 1.00` with
reward at the floor and `tok` at the cap is truncation (raise the cap), `tied 1.00` with
reward at the ceiling is saturation (harder task) — 55's curve had 3 of the first and 7 of the
second in one run, and the `tied` field alone cannot separate them. **Even a passing value run
only shows the paths executed with effect and did not crash**; a wrong seed stride or a wrong
mask trains worse without going red on a card. Correctness rests on the three CPU mutants.

*Two failures before the first value step, both costing a full attempt:* the bare hub id
`Qwen/Qwen3-27B` cannot resolve on this pod (no network — the weights are at
`/work/Qwen3.8-27B-NVFP4`), and the fp4 GEMM does not codegen under the pod's default
`python3` (tilelang 0.1.8): 8 × `no instance of overloaded function "tl::tma_load"`. Both are
environment, not this change; the working peer runs put `/work/tl013/bin` (0.1.13) on `PATH`
and set `TILERL_QWEN38_SOURCE`. A one-step smoke test costs 4 minutes and would have caught
both.

The card arm needs the training engine sized for `prompts_per_step × group` rows. That sizing's
length axis was a separate open defect when this was written and landed meanwhile (#322, per
consumer rather than per axis), so rebasing onto it makes the row axis here the only change.
Verified across the three 8-row splits at `--eval-max-new-tokens 2048`: 1×8, 2×4 and 4×2 all
size to **1328 blocks**, which is the point — the same rows at the same lengths must cost the
same pool, so the comparison is not confounded by memory.

Both arms must run in one sitting on one revision. Measuring 1×8 now and 2×4 after the
intervening changes would trade a cap confound for a version confound.
