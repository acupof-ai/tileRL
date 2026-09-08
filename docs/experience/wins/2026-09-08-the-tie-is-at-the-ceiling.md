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

| population | ratio | pooled idle | backfill ceiling 1/ratio |
|---|---:|---:|---:|
| GSM8K (48's probe, same 6144 cap) | 0.414 | 0.586 | 2.42x |
| **level 5, all problems** | **0.728** | 0.272 | 1.37x |
| **level 5, no sample at cap (26 of 34)** | **0.696** | 0.304 | **1.44x** |

The cap-free and all-in figures differ by **0.032**, so clamping is not what makes level 5 look
homogeneous — the ratio is real. Stable across n=16/23/34, no drift with sample size.

**Consequence for two tail levers, both priced on GSM8K.** Backfill measured 1.53x end-to-end
against its own 2.42x ceiling, a 37% capture; the same capture on level 5's 1.44x ceiling is
**1.19x — derived, not measured** (capture ratio 0.374 from GSM8K, level-5 cap-free ratio 0.696,
n=26). Train padding moves the same way. **The two tail-driven levers largely disappear on the
candidate task**, leaving batch (1.31x) and kernel work. That is a re-ranking, not a correction.

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
priced on one task needs `1/ratio` recomputed on the task it will run on — and the ratio needs
its population stated, since a cap flattens it toward 1.

## Results

| date | commit | machine | target | model | n | k | base | tied | usable | ratio (cap-free) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-09-08 | 2f25f26 | H20 ×4 | cuda | Qwen3.8-27B NVFP4 | 34 | 8 | 77.2% | 67.6% | 32.4% | 0.696 |

Raw artifacts: `/work/pk_{0,25,50,75}.jsonl` (one JSON row per problem: `correct`, `tokens` per
sample, `at_cap`, `distinct`), `/work/pk{A,B,C,D}.log`.
