# The killed eval arm kept nothing — 2026-09-09

## Context

The MATH level-5 before-arm was killed 1h40m into a 500-question eval. The run
dir held `manifest.json` + `provenance.txt` only: zero scored rows on disk.
The same day, a second run was killed mid-eval with the same result. The eval
arm is the longest segment of a run — the before-arm alone was 32% of
gate-to-gate on the run that measured it — so it is the segment most likely to
be killed, and each kill cost the whole arm.

## Root cause

`generate_ids` buffered every completion in an in-memory list and returned only
when all prompts finished. `gsm8k_accuracy` decoded and scored after the return,
and `_write_eval_rows` wrote the whole file with mode "w" at the end. Three
buffers in series, one write at the end: interruption cost = everything.

## Fix

`generate_ids` takes an optional `on_row(i, ids)` fired in the poll loop as each
completion lands; `gsm8k_accuracy` scores inside the callback and forwards the
row; `cli._eval_row_appender` appends one JSON line per row with open/close per
row, so a kill loses at most the row in flight. A killed GSM8K arm or curve
point now keeps every row that finished. The MMLU arm is not covered:
`mmlu_accuracy` has no `on_row`, so it still lands whole and a kill mid-MMLU
still loses it. Rows land in COMPLETION order, so `curve_churn` was switched
from position pairing to the `i` key (the other readers already keyed on it).
Overhead: 9.2 µs per arm on a stub engine (n=1000), 9 ns per row —
wins/2026-09-09-eval-arms-write-incrementally.md.

## Rule

长时间的 eval 臂必须增量写盘,否则它的中断代价是全部。
