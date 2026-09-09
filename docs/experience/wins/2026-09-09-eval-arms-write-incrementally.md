# Eval arms write rows as they land — 2026-09-09

## Context

The MATH level-5 before-arm was killed 1h40m into a 500-question eval, and the run
dir held only `manifest.json` + `provenance.txt` — zero scores, zero completions.
`generate_ids` buffered every completion in memory and the only writer
(`_write_eval_rows`) ran in the caller after the whole arm. The eval arm is the
longest segment of a run (the before-arm alone was 32% of gate-to-gate on the
run that measured it), so it is the segment most likely to be killed — twice
today alone, both times mid-eval.

## What worked

- `generate_ids` takes an optional `on_row(i, ids)` fired in the poll loop as each
  completion lands, in COMPLETION order. Default None: one `is not None` per tick.
- `gsm8k_accuracy` scores each row inside the callback and forwards it to its own
  `on_row`; `per_problem` still lands in prompt order (sorted by `i`).
- `cli._eval_row_appender` appends one JSON line per row to `eval-<tag>.jsonl`
  (open/close per row, so a kill loses at most the row in flight). The GSM8K arm
  and every curve point stream through it. The before-arm cache payload is
  unchanged: it snapshots the in-memory rows, which still carry both arms.
- **Coverage is GSM8K, not MMLU.** `mmlu_accuracy` has no `on_row`, so the MMLU
  arm still buffers and lands whole — a kill mid-MMLU still loses that arm
  (about 12% of before/after wall time: 208.3 s MMLU vs 1452.9 s GSM8K on the
  run that measured both). The fix covers the arm that was 1h40m, not every
  eval arm.
- `curve_churn` now pairs by the `i` key, not file position: the file is in
  completion order since today, and position pairing would silently compare
  different questions. `paired_se` and `_mcnemar` already keyed on `i`.

Overhead, measured on a stub engine (n=1000, concurrency 8, 20 runs each):
the `on_row=None` path is 272.2 µs per arm; the callable path 281.5 µs — a
9.2 µs delta, 9 ns per row. On a real arm the poll loop is a negligible
fraction of wall time (the stub loop IS the whole cost here), so the added
branch is bounded above by 9.2 µs per arm. The selftest in `eval.py`
(`__main__`, no args) drives a reverse-completion engine and goes red on an
index-by-completion bug; `tests/test_eval_rows.py` drives the callback path.

## Rule

A long eval arm writes rows as they land; its interruption cost is one row, not
the arm. Any reader of an eval jsonl pairs on the `i` key — file order is
completion order, never position.
