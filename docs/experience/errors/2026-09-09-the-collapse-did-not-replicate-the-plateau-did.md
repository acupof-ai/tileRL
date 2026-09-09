# The collapse did not replicate; the plateau did — 2026-09-09

**Status:** the step-75 collapse of run `86a06dc8c420` (−11.0 pt, 6.62σ) was
**non-deterministic** — a seed-1 replication (run `d183f74d233c`, identical
config, `--seed 1`) scored 94.4 / 94.2 / 94.8 / 94.8 at steps 25/50/75/100 with
no dip anywhere. The reproducible feature of both runs is not the collapse but
the **plateau**: the score arrives by step 5-25 and the remaining training buys
nothing. The early-stop case rests on the plateau, and the plateau is what the
pricing below values.

## Context

The collapse question (systematic vs one-off) decided whether early stopping is
mandatory. The replication used the same recipe, grid, and n=500 curve; the
only changed input was the seed. A hypergeometric overlap test (cc's instrument)
checked whether the two seeds lost the same questions at step 75 — it became
unnecessary when seed 1 lost nothing.

## What the two seeds say

| step | seed 0 | seed 1 |
|---|---|---|
| base | 87.4% | 87.4% |
| 5 | 94.2% | — |
| 10 | 94.2% | — |
| 15 | 94.6% | — |
| 20 | 92.8% | — |
| 25 | 93.2% | 94.4% |
| 50 | 93.4% | 94.2% |
| 75 | 82.4% | 94.8% |
| 100 | 91.2% | 94.8% |

**The collapse is non-deterministic, and its rate is unmeasurable.** One
collapse in two runs gives a 95% interval of roughly [0.013, 0.987] — it
excludes "every run collapses" and nothing else. It does not justify "rare".
Distinguishing p<5% from p>20% needs more runs than the compute buys, so the
stop decision must not depend on the rate — and it does not need to.

**The plateau is the finding both seeds share.** Seed 0 reached its best score
(94.6%) at step 15; seed 1 was at plateau by step 25. Running the full 100
steps cost 17.4x the training time of step 5 and finished 1.4 pt *lower* than
the step-5 score. Every post-plateau step is pure cost.

## Pricing an early-stop configuration

The wall clock is eval-bound, not train-bound: seed 1 spent 3183 s in four
curve evals against 2067 s training (62% eval). Densifying the grid at n=500 is
infeasible (eval-every 5 ≈ 16000 s of eval). The lever is `--eval-curve-n`.

Replaying both curves with raw-score patience=1 (strict improvement resets;
stop at the first non-improving eval; 20 s/step training; 800 s per n=500 eval,
linear in n — seed-1 measured average 796 s):

seed 0, fine grid (full run 3600 s):

| curve-n | stops at | retained (best step) | total | vs full |
|---|---|---|---|---|
| 500 | step 10 | 94.2% (step 5) | 1800 s | 2.0x |
| 100 | step 15 | 94.2% (step 10) | 780 s | 4.6x |
| 50 | step 10 | 94.2% (step 5) | 360 s | **10.0x** |

seed 1, 25-step grid (full run 5250 s):

| curve-n | stops at | retained (best step) | total | vs full |
|---|---|---|---|---|
| 500 | step 50 | 94.4% (step 25) | 2600 s | 2.0x |
| 100 | step 50 | 94.4% (step 25) | 1320 s | 4.0x |
| 50 | step 75 | 94.2% (step 50) | 1740 s | 3.0x |

(Retained scores are full-500 scores at the best step. Subset scores at
n=100/50 run ±1.6 pt off the full score — the n=100 subset itself scores 89.0%
at base against 87.4% full — so a retained score must be confirmed by the n=500
anchor, never read off the curve subset.)

## The failure modes, priced

**Raw-score patience is fitted to these curves, not derived.** It works here
because both curves are step functions: the entire gain lands in the first
eval (seed 0: +6.6 pt at step 5, then flat; seed 1: +7.0 pt by step 25, then
flat), so "stop at the first non-improving point" never gives anything back.
On a gradually-rising curve the same rule stops at the second point and
discards every later gain. Worse, its trigger is below the instrument's own
floor: seed 1's n=500 stop fired on 94.2% ≤ 94.4% — a one-question difference,
smaller than the 0.4 pt net floor measured below. It is acceptable here
only by asymmetry — with best-weight keeping, stopping early risks future
gains, never the score already in hand. That asymmetry is a property of this
task, not of the rule.

**Significance-gated patience is unsafe at n=100.** The base→step-5 jump is
+6.6 pt. Measured paired 2×SE at n=100: 5.6 pt on seed 0's pair (detectable,
1.07x), 7.4 pt on seed 1's base→step-25 (not detectable, 0.81x). At n=50:
6.8 pt against a +6.0 pt signal — dead. A patience rule that waits for a
*significant* improvement would sometimes stop at step 5 on noise. This is the
mode #334 (cd06b2a) implements; a raw-score mode does not exist in the tree
yet.

**n=50 goes blind.** Consecutive plateau points often differ on 0 of the 50
subset questions; the paired SE is then 0 and the instrument cannot see any
movement smaller than ~2 questions. The 10x row is real but its stop step is
noise-dominated (±5 steps across plausible subsets).

**Early stop and best-weight keeping are coupled.** Without an adapter-best
checkpoint, the retained weights are the *stop* step's, not the best step's:
seed 1 at n=500 stops at step 50 (94.2%) while the best step scored 94.4%.
Selling early stop without best-weight keeping loses the margin.

**The eval-cost model is optimistic for the fine grid.** The fine-grid run
measured 925-1135 s per n=500 eval (avg 1026 s), 28% above the 800 s model.
Real prices at n=100 are ~205 s/eval, not 160 s.

## The eval floor, measured twice

The score-level floor is ±2 questions (0.4 pt): the after-arm and the
step-100 curve point scored the same step-100 adapter on the same 500
questions, both greedy, and got 472 vs 474. The net hides the gross — **52 of
the 500 questions (10.4%) flipped between the two passes**, 27 one way and 25
the other. Completion lengths diverged (median ratio 1.66x, max 9.9x; 18 of 52
by more than 2x), in both directions. This is not a fixed set of boundary
questions; it is batch-composition non-determinism (the after-arm batches in
file order alongside MMLU; the curve point shuffles and batches GSM8K
alone). Pinning the questions does not fix it; pinning the batch composition
would reduce it, with JIT and numeric noise still left.

The sharp consequence: the same adapter re-evaluated flips 52 questions; the
step-75 and step-100 adapters, 25 training steps apart, flip 50. **On this
instrument, same-policy re-eval noise equals the apparent per-question
difference between policies 25 steps apart.** Per-question plateau signal is
entirely inside the eval noise; only net scores (stable to ±0.4 pt) carry
signal. Every cross-run per-question comparison — the overlap instrument
included — sits on this 10% flip rate.

The earlier floor reading (1 question, 0.2 pt) was a single measurement; two
readings now bound the net floor at 0.4 pt, and the gross flip rate is
measured for the first time.

## The economics turn on the stop point

A fine grid looks expensive — eval-every 5 implies 20 evals — but patience=1
stops at step 10, so only 2-3 evals are ever paid. **The eval bill is set by
the stop point, not by the budget.** The dense grid and the early stop are
one mechanism: the grid is affordable precisely because the stop makes most
of its points never run.

## What this does and does not authorize

The table above is an **offline replay** of two curves recorded under
`--eval-every 25 --eval-curve-n 500` with no patience. The candidate config
(`--eval-every 5 --eval-curve-n 100`, raw-score patience=1, best-weight
keeping) differs in three parameters, two of which change the curve's own
shape — sampling density and per-point noise. Pricing the new config off the
old config's records is the lever-priced-against-a-changed-configuration
error; the numbers above are a reason to run the config, not a measurement
of it. MMLU held across the run (75.1% → 76.6%) — a health check, not a gain.

The path: this change lands the measurement (this entry, the overlap-test
entry, `scripts/curve_table.py`). A following change adds
`--patience-mode {significant,raw}` (default significant, preserving cd06b2a).
A real run of the candidate config, priced from its own records, is what
earns the default flip.

## Rules

- **A non-replication proves "not deterministic", nothing about the rate.**
  1/2 occurrences has a 95% interval of [0.013, 0.987]; "rare" is not in it.
  When the rate is unmeasurable, build the decision so it does not depend on
  the rate.
- **Price the instrument that prices the claim.** Eval is 62% of this run's
  wall clock; the grid's resolution is bought in eval seconds, and the lever
  is the subset size, not the step interval.
- **A stop rule fitted to two curves is not a derived rule.** Raw-score
  patience works on step-function curves and below the eval's own floor;
  state the curve shape it assumes, or it will be read as universal.
- **The eval bill is set by the stop point, not the budget.** A dense grid is
  affordable when early stop makes most of its points never run; price the
  grid and the stop as one mechanism.
- **A net floor and a per-question floor are two instruments.** The net score
  is stable to ±0.4 pt while 10.4% of questions flip between two passes of
  the same adapter; quoting the net as "the floor" hides the noise every
  per-question comparison sits on.
- **A retained score belongs to the n=500 anchor, not the curve subset.**
  Small subsets run ±1.6 pt off the full score; reading the retained score
  off the curve would over-credit the cheap configuration.

## See also

[The step-75 dip hit problems a second seed solved at the same step](2026-09-09-the-dip-hit-problems-a-healthy-seed-solves.md)
— the dip's mechanism (72 of seed 0's 88 step-75 wrongs are solved by seed 1)
and the hypergeometric overlap instrument, deferred for a dip that
reproduces. This entry is what the plateau buys; that one is what the dip was.
