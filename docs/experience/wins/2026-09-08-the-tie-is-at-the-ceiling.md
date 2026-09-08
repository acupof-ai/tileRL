# The tie is at the ceiling, and level 5 is homogeneous — pass@k, 2026-09-08

> Status: pending-remote (n=100 in flight; figures below are n=34 of 100)

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

**One run yields three numbers that were being estimated separately.** n=34, k=8, cap 6144,
temperature 1.0, 27B NVFP4 on one H20, sharded across four cards:

| correct of 8 | problems |
|---:|---:|
| 0 | **4** |
| 1 | 1 |
| 2–3 | 0 |
| 4 | 3 |
| 5 | 2 |
| 6 | 0 |
| 7 | 5 |
| 8 | **19** |

- base **210/272 = 77.2%**
- **tied 23/34 = 67.6%** (19 at the ceiling, 4 at the floor)
- usable 11/34 = 32.4%

**The distribution is bimodal and asymmetric, and that asymmetry is the actionable part.**
19 problems at 8/8 against 4 at 0/8: the tie is at the **ceiling**. `--prompts-per-step` is the
lever that reaches it — 8 distinct problems tie with probability ∏pᵢ rather than one problem's
p⁸ — and it cannot reach the floor, which is a cap or difficulty question. Had the tie been at
the floor, that change would have been the wrong medicine.

**32.4% "usable" is the optimistic reading.** 5 of the 11 untied problems sit at 7/8, where the
group's std is small, the advantage is ≈±1/√7, and one step of learning promotes them to 8/8 and
out of the usable set permanently. The robust band is 4/8–5/8: **5 problems, 14.7%.**

## Length distribution: level 5 is homogeneous, and GSM8K may not be a fair comparison

`Σlen / Σ(max_in_group × k)` — 1.0 means every sample in a group is the same length, so nothing
waits:

| population | occupancy | pooled idle | backfill ceiling |
|---|---:|---:|---:|
| GSM8K (another session's probe, same 6144 cap, k=8) | **0.2547** | 74.5% | 2.17x |
| **level 5, all problems** | 0.728 | 27.2% | 1.39x |
| **level 5, no sample at cap (26 of 34)** | **0.696** | 30.4% | **1.42x** |

Both columns are `Σlen / Σ(max_in_group × k)` at k=8, where a group is one problem's k samples —
the GRPO group itself, so no regrouping was needed. **The first version of this table put 0.696
beside 0.414, which is `Σmax / Σsum`** — the same data inverted and missing the group factor, so
the two differed by 8x. `1/(0.4908 × 8) = 0.2547`. Both quantities are dimensionless and land near
0.5, so nothing about reading them says they are not the same measure. The corrected gap is **2.8x,
not 1.8x**. Occupancy also depends on k: doubling the group roughly halves it, since `max × k` grows
faster than `Σlen`.

The cap-free and all-in figures differ by **0.032**, so clamping is not what makes level 5 look
homogeneous — the ratio is real. Stable across n=16/23/34, no drift with sample size. Note the
sign: here excluding capped problems *lowers* occupancy, because a problem whose every sample hits
the cap has zero within-group spread (one such problem reads exactly 1.000). On GSM8K the same
exclusion moves it the other way, since there the capped rows are long *steps* whose idle was
already high.

**Consequence for two tail levers, both priced on GSM8K.** The ceiling is not `1/occupancy`:
`wall = a + b·max + c·sum` with measured `b = 8.83 ms/tick`, `c = 3.705 ms/row/tick`, and backfill
only removes the `max` term — `c·sum` is the per-row KV cost and survives it. So the ceiling is
`(b·(max/sum) + c)/c`, giving **2.17x on GSM8K** (against 1.67x measured over 20 steps — a 77%
capture) and **1.42x on level 5 at k=8, 1.21x at k=16**. Backfill on level 5 therefore prices at
**1.05–1.10x**, and **is not worth doing there**.

My first estimate of 1.19x was wrong three ways at once: a mid-run 1.53x instead of the final
1.67x, `1/occupancy` as the ceiling (which assumes tick cost is independent of occupancy, and `c`
is the counterexample), and a capture ratio derived by dividing those two wrong numbers. **The two
tail-driven levers largely disappear on the candidate task**, leaving batch (1.31x) and kernel
work — a re-ranking, not a correction.

**And the GSM8K half is pending.** 28 of 272 level-5 samples reached the 6144 cap (10.3%) against
GSM8K showing a row ≥5000 tokens in 12 of 18 steps (66.7%) — **2.9x more, on the easier task with
100–150-token reference answers.** That asymmetry rules out "the model is simply verbose at this
cap", since the harder task would then truncate more, not less. Degeneration on GSM8K is under
test by another session; if confirmed, **0.414 is a defect's fingerprint rather than a task
property**, and the right move is to drop that half of the comparison rather than reprice it.

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

**Which end the tie sits at picks the fix.** Ceiling ties yield to more prompts per step; floor
ties do not, and no amount of regrouping helps a problem the policy cannot solve.

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
| 2026-09-08 | 2f25f26 | H20 ×4 | cuda | Qwen3.8-27B NVFP4 | 34 | 8 | 77.2% | 67.6% | 32.4% | 0.696 |

`occupancy` is `Σlen/Σ(max·k)` over problems with no sample at the cap, k=8, groups being the
problems themselves.

Raw artifacts: `/work/pk_{0,25,50,75}.jsonl` (one JSON row per problem: `correct`, `tokens` per
sample, `at_cap`, `distinct`), `/work/pk{A,B,C,D}.log`.
