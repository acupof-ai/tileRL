# GSM8K rollout lengths on the 27B: cap 1024 does not bind, p90 is 532

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55
**Card:** H20 card 5, one process, `tilerl-gsmlen`, tree `/work/tilerl-s-v100-sm70-fp4` sha 268dff4

## Context

The first ISO-RL arm read `tied 100%` because it ran at `--max-new-tokens 128` — a cap taken
from a recorded 1029.1-token figure that is MATH's, not GSM8K's. The `steps_to_score` curve
needs this task's own cap, and a cap chosen from the wrong task produces a run where every
group ties at the floor and no gradient flows.

## What was measured

`scripts/probe_gsm8k_lengths.py`, 32 prompts x group 4 at temperature 1.0, cap 1024,
thinking off, three seeds. Non-thinking rollouts, `render_chat` + `prompt.sampling` — the same
two calls `eval.py:146` and `cli.py:626` use, so the probe's input is the policy's.

| | seed 0 | seed 1 | seed 2 | pooled (n=384) |
|---|---:|---:|---:|---:|
| accuracy | 94.53% | 96.09% | 95.31% | 95.31% |
| mean | 322.5 | 315.7 | 327.7 | **322.0** |
| median | 279 | 253 | 287.5 | 274 |
| p90 | 512.5 | 530.2 | 534.6 | 532.2 |
| p99 | — | — | — | 1024.0 |
| max | 1024 | 1024 | 1024 | 1024 |
| at cap | 1.6% | 0.8% | 1.6% | **1.3%** |
| s/rollout | 5.41 | 4.75 | 5.00 | 5.05 |

**A cap of 1024 does not bind** at 1.3% touching it. p90 = 532, p99 = 1024.

Cross-seed accuracy sd is **0.78 pt** over 3 seeds at 128 rollouts each. This is the noise on
the quantity GRPO differences within a group, so it prices `--group`: a per-step reward move
under 0.78 pt is sampling noise. It is **not** the reward-vs-step curve's error bar —
`gsm8k_accuracy` scores at `temperature=0.0` (`eval.py:147`) and `reference.py:1220` takes
argmax there, so the curve's eval is deterministic given weights and its width is the binomial
SE on the subset size (`ledger.py:127`).

## What decided it

`--max-new-tokens 1024` for the `steps_to_score` arms, unchanged. The alternative worth
considering was 640 (p90 + margin), which would save decode time on the 98.7% of rollouts that
finish early — it saves nothing, because a rollout that finishes emits `<|im_end|>` and the
engine stops it; the cap only costs time on the 1.3% that reach it.

## Instrument check

Two numbers agree with an independent measurement made four days earlier under a different
protocol: mean **322.0 against 320.8** (0.4%) and median **274 against 283**, from the
uncapped greedy n=40 table in
[errors/2026-09-04-the-eval-cap-measured-itself.md](../errors/2026-09-04-the-eval-cap-measured-itself.md).
Different split (train vs test), different temperature (1.0 vs 0), different session. The
length distribution is the policy's.

Accuracy is where the disagreement is, and it is not resolvable. 95.31% sits +5.71 pt above
the measured 89.6% base (n=500, full test split). The comparison's n is the **prompt** count,
32, not the 384 rollouts — four samples of one prompt are correlated — giving SE 3.98 pt and
z = 1.44. Not significant, and the two candidate explanations (train-vs-test split, and 32
prompts being a narrow slice) are both larger than the effect. Nothing here needs the
instrument to be wrong.

## Rule

A cap is a property of a task and a thinking mode, not of a model. Read it off the
distribution before an arm runs, and report the cap-hit rate beside it: the same table that
says "p90 is 532" says whether the number you are looking at is the policy's or the cap's.
