# Training runs record their own GSM8K arms to the bench store — sm90, 2026-09-09

> Status: Shipped (emission code; first rows pending the next pod training run)

## Context

The README's headline training number is GSM8K: base 89.6% → 94.8% after 100
GRPO steps. The ruler registry had no `gsm8k_pct` — the acceptance metric for
the RL line was unmeasurable by the system built to measure everything else.
The reverse audit (README number → registry metric) found this gap and three
smaller ones (rollout tokens, solve wall, train-prompt solve rate — reported,
not yet registered).

## What Worked

The eval arms already run inside every training run (`_train_adapters`,
before/after). The number exists there and nowhere else, so the collector is
the run itself: after each GSM8K arm, `_emit_gsm8k_record` appends a validated
`gsm8k_pct` row to `docs/experience/bench/measurements.jsonl`.

- Population is `shape.steps` (0 = before arm, N = after N steps) — before and
  after are two rows, never one overwriting the other.
- `spread` is the binomial SE of the proportion (`sqrt(p(1-p)/n)`); `n` is the
  question count. A proportion measured on 200 questions carries its own
  dispersion — no rerun needed.
- `warm.compiles = 0` is exact, not an engine assertion: JIT time can enter a
  seconds figure, not a proportion.
- Floor is `measured-best` — the regression direction for an acceptance metric
  is "training made it worse".

The emitter reaches `scripts/benchrec.py` through the same
`Path(__file__).parents[2]` bridge `cmd_bench` uses, so every row passes the
store's validator (a bad row raises at the training run, never reaches the file).

## Rule

A metric whose only measurement site is inside a training run is collected by
that run — not by a separate script that re-loads the checkpoint. The
measurement happens once, at the source.
