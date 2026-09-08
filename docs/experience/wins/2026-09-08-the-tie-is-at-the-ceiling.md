# The tie is at the ceiling, and level 5 is homogeneous — pass@k, 2026-09-08

> Status: final, n=100 of 100.

## Context

Task selection for P1 had been done on **difficulty** — base accuracy — and that is a proxy.
What decides whether GRPO produces a gradient is whether a group's rewards *differ*: a problem
the policy gets 3 of 8 times carries advantage, one it gets 8 of 8 or 0 of 8 is a constant no
matter how the dataset is labelled. GSM8K failed P1 by being solved (88.0% base, 73 of 81 steps
tied at the ceiling), MATH level 5 was chosen as its harder successor, and then level 5 measured
**91.0%** once its cap stopped scoring truncated correct answers as wrong
(`errors/2026-09-08-a-cap-reported-as-a-base.md`) — **easier than the task it replaced.**

So the quantity was never in the ledger. `scripts/pass_at_k.py` measures it directly: k samples
per problem, one row per problem, distribution reported rather than folded into an accuracy.

## What Worked

**One run yields three numbers that were being estimated separately.** n=100, k=8, cap 6144,
temperature 1.0, 27B NVFP4 on one H20, sharded across four cards:

| correct of 8 | problems |
|---:|---:|
| 0 | **6** |
| 1 | 4 |
| 2 | 2 |
| 3 | 0 |
| 4 | 5 |
| 5 | 5 |
| 6 | 4 |
| 7 | **15** |
| 8 | **59** |

**And the tied fraction this yields is an interval, not a number.** 65 of 800 samples reached the
6144 cap (8.1%, across 24 rows), and a truncated sample is unscored rather than wrong — so each
row's `correct` is a lower bound and `correct + at_cap` an upper one:

| reading | base | tied | ceiling | floor |
|---|---:|---:|---:|---:|
| as measured (truncation scored wrong) | 81.8% | **65.0%** | 59 | 6 |
| upper bound (truncation scored right) | — | **82.0%** | 77 | 5 |
| cap-free subset, n=76 | — | — | — | — |

The third row is deliberately empty: selecting `at_cap == 0` selects short completions, which
selects easy problems, which selects 8/8, so that subset is **a third population rather than a
cleaner read of the first**. On the n=70 partial it read 79.2% against an 80.0% upper bound, and
that near-agreement is not corroboration — both estimates are biased the same way, so their
agreement is evidence of the shared bias, not of the value.

The upper bound's precondition was checked rather than assumed: **0 rows have
`correct + at_cap > k`**, so no capped sample was ever scored correct and truncation does imply
unscored here. Had that count been non-zero the upper bound would be wrong while still looking
right. The degeneracy guards are clean too — **0 rows with `distinct == 1`, 0 zero-length
completions** — so the spread is the sampler's and not an artifact of a seed that failed to vary.

**What survives the 17.0-point width is the direction, and that is what the lever needs.**
Both ends put the tie at the **ceiling** — 59 against 6, 77 against 5. `--prompts-per-step` reaches
that end and cannot reach the floor, which is a cap or difficulty question. Had the tie been at the
floor, that change would have been the wrong medicine, and no reading of the interval moves it
there. **How it reaches the ceiling is not the obvious way**: the first framing was "8 distinct
problems tie with probability ∏pᵢ rather than one problem's p⁸", and that needs one sample per
problem — which GRPO cannot do, since its baseline is the group mean and a one-sample group has no
defined advantage. The real shape is `group = prompts × completions` with the tie still computed per
prompt, so fewer completions makes **each group tie more often** and the entire gain comes from a
step having several groups, only one of which must be untied. Opposite direction, different
mechanism, same conclusion — see the pricing below.

**And the deeper reason the direction is what matters:** the 8 samples come from *one* problem, so
a ceiling tie is a statement about **within-group correlation**, not about the task being too easy.
A harder dataset does not fix it — a hard problem sampled 8 times ties at the *floor* and carries
no gradient either. Difficulty is a property of the task; tie mass is a property of this model's
hesitation on it, and the two were being treated as one quantity when level 5 was selected.

**The magnitude does not survive, and a three-band decision criterion consumes the magnitude.**
65.0% and 82.0% fall in different bands of the criterion a peer built on the first point estimate,
so this reports the interval rather than a band. The width is systematic, not statistical: n=70 →
n=100 moved the lower bound 61.4% → 65.0% (+3.6 pt against SE 5.7, z=0.63 — noise) while the width
barely moved, 18.6 → 17.0. **More problems shrink only the sampling error.** Only a rerun of the 24
contaminated rows at a higher cap narrows it, and the agreed plan is to fold that into the post-
`--prompts-per-step` re-measurement at **cap 12288** rather than spend four cards on it now.

**"Usable" is the optimistic reading twice over, and more so at n=100.** 15 of the 35 untied rows
sit at 7/8, where the group's std is small, the advantage is ≈±1/√7, and one step of learning
promotes them to 8/8 and out of the usable set permanently. The robust 3/8–5/8 band is **10
problems, 10%** — against 35% nominal usable.

**The floor is difficulty, not degeneration.** Of the 6 rows at 0/8: **4 have `at_cap = 0`,
`distinct = 8`, lengths 369–1788** — the model finished and was wrong; **1 is 8/8 capped**
(6144–6144, the cap's verdict rather than the policy's); **1 is mixed** (i=92, 3 of 8 capped,
4814–6144). So `--prompts-per-step` genuinely cannot help that end, but nothing there is a loop.

## What the distribution prices: `--prompts-per-step` is 1.21x, not a cure

The lever this measurement was taken to justify can be priced from the same rows, because the 8
samples of a problem are **exchangeable** — `correct` is a sufficient statistic, so a group of
`comp < 8` is obtained by drawing `comp` of the real samples without replacement. No model of
per-problem difficulty is needed.

A step is gradient-free only if **every** prompt's group ties, and prompts are drawn independently,
so `step_tied = q_comp ^ prompts`:

| split | per-group tie `q_comp` | step gradient-free | steps with gradient | vs 1×8 |
|---|---:|---:|---:|---:|
| 1 × 8 (today) | 0.6496 | 0.6496 | **35.0%** | 1.000x |
| **2 × 4** | 0.7621 | 0.5807 | **41.9%** | **1.197x** |
| 4 × 2 | 0.8715 | 0.5769 | 42.3% | 1.208x |

**2×4 captures essentially all of it** — per-group tie rises almost as fast as the group count, so
4×2 buys 0.4 more points. And the ceiling is **1.21x**, an order of magnitude short of what the
65% tie fraction suggests on sight: removing a tie from a group does not remove it from the step.

**The measured `q_comp` matters, and the first pricing used an interpolated one.** Substituting
`p_i = correct_i/8` into `E[p^comp + (1-p)^comp]` gives 0.7229 / 0.7955 / 0.8875 — high at every
`comp`, because a problem observed at 8/8 has a true `p < 1` and the interpolation reads it as 1.
That was noted as a bias with the argument that a *ratio* would survive it. It does not: the bias
is **−0.073 / −0.033 / −0.016**, shrinking monotonically in `comp`, so it inflates the numerator
more than the denominator and the ratio comes out **1.37x instead of 1.21x**. A bias that varies
with the quantity it biases does not cancel in a ratio, and "all three are high in the same
direction" is not sufficient for it to. The `comp = 8` row is the control: measured 0.6496 against
this run's 0.650 tied fraction, agreeing to 0.0004.

`q_comp ^ prompts` was checked rather than assumed — `E[X]^g ≠ E[X^g]` in general, and these
problems are wildly heterogeneous. An empirical resample of actual problems gives 0.6337 against
the formula's 0.6328. It holds because prompts are drawn *independently*: heterogeneity inflates
the variance of the estimate, not its expectation.

**And the re-measurement's prediction is fixed here, before the change lands.** At cap 12288 with
2 × 4: per-group tie **≈ 0.762**, step gradient-free **≈ 0.581**. A measured per-group tie
materially above 0.80 means the prompts in a step are not independent (correlated difficulty within
a batch); between 0.76 and 0.80 is the gap between the measured and interpolated estimators and is
not a failure. One confound the prediction cannot resolve: raising the cap to 12288 also un-ties
floor problems (2 of the 6 here have capped samples), which lowers tie for a reason unrelated to
the split — so the re-measurement must run **1×8 at 12288 as well**, or the two effects arrive as
one number. **And both arms have to run in one sitting on one revision.** Measuring the 1×8 baseline
now and the treatment after the change would trade a cap confound for a version confound —
`group_advantages`, the empty-rollout fix and the pool fix all land in between.

## Length distribution: level 5 is homogeneous, and GSM8K may not be a fair comparison

`Σlen / Σ(max_in_group × k)` — 1.0 means every sample in a group is the same length, so nothing
waits:

| population | occupancy | pooled idle | backfill ceiling |
|---|---:|---:|---:|
| ~~GSM8K (another session's probe, same 6144 cap, k=8)~~ | ~~0.2547~~ | ~~74.5%~~ | **withdrawn** |
| **level 5, all problems** | 0.706 | 29.4% | 1.33x |
| **level 5, no sample at cap (76 of 100)** | **0.679** | 32.1% | **1.34x** |

Both columns are `Σlen / Σ(max_in_group × k)` at k=8, where a group is one problem's k samples —
the GRPO group itself, so no regrouping was needed. **The first version of this table put 0.696
beside 0.414, which is `Σmax / Σsum`** — the same data inverted and missing the group factor, so
the two differed by 8x. `1/(0.4908 × 8) = 0.2547`. Both quantities are dimensionless and land near
0.5, so nothing about reading them says they are not the same measure. The GSM8K row is **withdrawn**, and not
because of that: its probe called `tok.encode(text)` instead of `render_chat`, so the model was
given a bare document and ran to the cap on prompts a chat template would have ended. Every GSM8K
tail figure from that path goes with it — pooled idle 74.5%, backfill 1.54x, 12 of 18 steps at the
cap. The number is not repriced, it is removed: a measurement taken off the production path is not
evidence with a wider error bar. **So the cross-dataset gap this table existed to state is
unmeasured**, and level 5's 0.679 is the only occupancy here on a correct path. Occupancy also
depends on k: doubling the group roughly halves it, since `max × k` grows
faster than `Σlen`.

The cap-free and all-in figures differ by **0.032**, so clamping is not what makes level 5 look
homogeneous — the ratio is real. Stable across n=16/34/70/100, no drift with sample size. Note the
sign: here excluding capped problems *lowers* occupancy, because a problem whose every sample hits
the cap has zero within-group spread (one such problem reads exactly 1.000). On GSM8K the same
exclusion moves it the other way, since there the capped rows are long *steps* whose idle was
already high.

**The cost model still holds; the coefficient it was calibrated on does not.** The ceiling is not
`1/occupancy`: `wall = a + b·max + c·sum`, and backfill only removes the `max` term — `c·sum` is the
per-row KV cost and survives it. So the ceiling is `(b·(max/sum) + c)/c`, and since
`max/sum = 1/(occupancy·k)`, a task's occupancy sets it. **The 1.34x this gave for level 5 is
withdrawn along with it**: `b/c = 1.813` was back-solved from the GSM8K arm, so it inherits that
arm's defect. What survives is the *shape* — that the terms a lever cannot remove set the floor of
what it can buy, and that a lever priced on one task needs its ceiling recomputed on the task it
will run on. Backfill on level 5 is **not priced** until a correct-path GSM8K run recovers `b/c`.

The k=16 figure the first version of this entry carried (1.21x) is withdrawn rather than restated.
It came from "occupancy roughly halves when k doubles", and that shorthand is *exactly* the
statement that `1/(occupancy·k)` is invariant — so it cannot also move the ceiling. The true
direction is that group max grows sublinearly in k while `Σlen` grows linearly, so `max/sum` and the
ceiling both fall; the magnitude needs a k=16 run, which has not been done.

My first estimate of 1.19x was wrong three ways at once: a mid-run 1.53x instead of the final
figure, `1/occupancy` as the ceiling (which assumes tick cost is independent of occupancy, and `c`
is the counterexample), and a capture ratio derived by dividing those two wrong numbers. That estimate is superseded twice over now, so no
number from it stands.

**And the third defect is the one worth carrying.** The first estimate was wrong by arithmetic; the
GSM8K arm is wrong by *instrument*, and no amount of care in the arithmetic would have caught it.
The sampler had no chat template, exactly as this run's first attempt had no `stop_token_ids` —
both produced completions that looked like completions, in the right units, in the plausible range.
Five void figures came out of those two defects in one day across four sessions, and **none of them
resembled each other**, so no one's number served as anyone else's alarm.

**The asymmetry that flagged it, and what it turned out to be.** 65 of 800 level-5 samples reached
the 6144 cap (8.1%) against GSM8K showing a row ≥5000 tokens in 12 of 18 steps (66.7%) — 8.2x more,
on the *easier* task with 100–150-token reference answers. That ruled out "the model is simply
verbose at this cap", since the harder task would then truncate more, not less, and this entry
predicted the GSM8K figure was a defect's fingerprint rather than a task property. **It was, and it
was not degeneration**: the probe called `tok.encode(text)` where the production path calls
`render_chat`, so the model received a bare document with no turn to end and wrote until the cap.
The prediction was right about the conclusion and wrong about the mechanism — worth recording,
because "the model loops" was the hypothesis being tested and a missing template produces the same
histogram.

## Two instrument defects this measurement had first

**Every sample ran to the cap on the first attempt — 64 of 64.** `SamplingParams.stop_token_ids`
defaults to `()`, and the place that fills it from the tokenizer is `prompt.sampling()`, so
constructing the params object by hand yields a sampler that can never stop. The tie fraction it
produced (62.5%) looked like a finding. Caught only by comparing against the eval path, which put
3 of 32 at the cap on the same problems. **The same default cost two other sessions a void probe
the same day** — one reported idle 0.0%, one got a `Σlen = group × gen` identity — so the fix
belongs in the constructor, not in three guards.

**And an idle fraction I computed on a sorted list read 15.0% where the engine's arrival order
gives 40.7%** — 2.7x from the ordering alone, and sorting by length before grouping is one of the
*fixes* for tail idle, so the number reported as the status quo was the post-fix one
(`errors/2026-09-08-two-formulas-on-the-wrong-population.md`).

## Rule

**Select a task on the reward distribution, not on accuracy.** Accuracy is one moment of it, and
the moment that decides whether GRPO learns is the mass at 0/k and k/k. A dataset labelled harder
can be tied *more*: level 5 is 91.0% where GSM8K is 88.0%.

**Which end the tie sits at picks the fix, but the fix's mechanism is not the obvious one.**
Ceiling ties yield to more prompts per step; floor ties do not, and no amount of regrouping helps a
problem the policy cannot solve. And regrouping does not work by making a group harder to tie —
GRPO's baseline is the group mean, so completions cannot drop to one — it works by putting several
groups in a step when only one needs to be untied. Each group ties *more*; the step ties less.

**A bias with a consistent sign does not cancel in a ratio unless its magnitude is proportional.**
The interpolated `q_comp` was high at every `comp` and the ratio was still wrong by 0.16x, because
the bias shrank monotonically in `comp` — largest on the numerator. "All the terms are biased the
same way" is an argument about sign; a ratio needs one about magnitude.

**A length-distribution figure is a property of a dataset, not of the model.** Every tail lever
priced on one task needs its ceiling recomputed on the task it will run on — from the cost model,
not from `1/occupancy`, since the terms a lever cannot remove set the floor of what it can buy.

**And an occupancy figure needs its direction, its k, and its population.** `Σmax/Σsum` and
`Σlen/Σ(max·k)` are the same data and differ by a factor of k; both are dimensionless and land near
0.5, so a comparison of the two survives inspection. Occupancy falls as k rises. And a cap
contaminates it in *either* direction, measured the same day: on level 5, clamping the samples of
one problem flattens within-group spread and lowers idle by 3 points; on GSM8K, clamping the long
steps preserves their already-high idle and raises it by 11.6 — same mechanism, two levels, two
signs. So "a cap pushes the ratio toward 1" is not a rule; "a ratio computed over capped samples
is not a property of the length distribution" is.

## Results

| date | commit | machine | target | model | n | k | base | tied | usable | occupancy (cap-free) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-09-08 | 2f25f26 | H20 ×4 | cuda | Qwen3.8-27B NVFP4 | 100 | 8 | 81.8% | 65.0–82.0% | 35.0% | 0.679 |

`occupancy` is `Σlen/Σ(max·k)` over problems with no sample at the cap, k=8, groups being the
problems themselves.

Raw artifacts: `/work/pk_{0,25,50,75}.jsonl` (one JSON row per problem: `correct`, `tokens` per
sample, `at_cap`, `distinct`), `/work/pk{A,B,C,D}.log`.
