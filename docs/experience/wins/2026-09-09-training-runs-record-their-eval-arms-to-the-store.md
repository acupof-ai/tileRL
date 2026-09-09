# Training runs record their own eval arms to the bench store — sm90, 2026-09-09

> Status: Shipped (emission code; first rows pending the next pod training run)

## Context

The README's headline training numbers are the RL table: GSM8K 89.6% → 94.8%,
mean completion tokens 1720 → 1253. The ruler registry had neither — the
acceptance metrics for the RL line were unmeasurable by the system built to
measure everything else. The reverse audit (README number → registry metric)
found the gap; it also produced one wrong mapping (351.8 read as wall seconds;
it is tokens/correct = 157601/448), caught by recomputing the column from its
operands.

## What Worked

The eval arms already run inside every training run (`_train_adapters`,
before/after). The numbers exist there and nowhere else, so the collector is
the run itself: after each GSM8K arm, `_emit_eval_records` appends two
validated rows to `docs/experience/bench/measurements.jsonl`.

- `gsm8k_pct` (weight 0.94): the acceptance verdict. Population is
  `shape.steps` (0 = before arm, N = after N steps) — before and after are two
  rows, never one overwriting the other. `spread` is the binomial SE of the
  proportion; `n` is the question count.
- `rollout_tokens` (weight 0.5, direction −): **mean** completion tokens per
  problem, not the total — the total encodes n in the value and is incomparable
  across runs with different question counts. `spread` is the per-problem SD.
- tokens/correct is NOT stored: it equals `rollout_tokens / gsm8k_pct` and is
  computed in the view. A stored ratio gets one chance to drift from its
  operands.
- `warm.compiles = 0` is exact, not an engine assertion: JIT time can enter a
  seconds figure, not a proportion or a greedy token count.
- Floors are `measured-best` — the regression direction for an acceptance
  metric is "training made it worse" (or longer, for tokens).

The emitter reaches `scripts/benchrec.py` through the same
`Path(__file__).parents[2]` bridge `cmd_bench` uses, so every row passes the
store's validator (a bad row raises at the training run, never reaches the file).

## Rule

A metric whose only measurement site is inside a training run is collected by
that run — not by a separate script that re-loads the checkpoint. And a
quantity derivable from two stored operands is a view, never a row. When
mapping a claimed number to a metric, recompute it from the table's other
columns — the unit error above survived reading, not arithmetic.
