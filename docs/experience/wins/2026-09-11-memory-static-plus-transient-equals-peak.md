# Occupancy closes: peak = Σ static rows + transient — CPU, 2026-09-11

> Status: Shipped (CPU tiny cell exact; GPU peak is `torch.cuda.max_memory_allocated`,
> first recorded on a card).

## Context

`stats()["memory"]` listed the named static allocations (weights, state slots, KV pool,
draft pool) with derived/measured columns, but the design's invariant — that the measured
resident PEAK is the named bytes plus everything unnamed — was only implied. Two
presentation surfaces (a `serve --dry-run` print loop and the running server's
`/health` stats) each shaped the rows differently, and occupancy never reached the bench
ledger where the kernel %bound already lives.

## What worked

- **Transient residual.** `memory.transient_bytes(rows, peak) = measured_peak − Σ static`
  (budget rows excluded); it raises when the peak is below the held rows (built too early
  or a row missing). `memory_table` appends the transient row and per-tier held totals,
  so `Σ static + transient == peak` is printed, not assumed.
- **One renderer.** `memory_table` (data) + `format_memory_table` (text) are the single
  surface. `/health` serves `engine.stats()["memory"]`; `serve --dry-run` builds the same
  table (with the budget rows its `device_free` implies). The engine measures the peak —
  `max_memory_allocated` on cuda, the held-storage sum on the CPU tiny cell, where
  transient is exactly zero.
- **One ledger.** `serve --dry-run --record-residency` appends a
  `device_resident_bytes` row (peak with its `static`/`transient` split in the shape) to
  `docs/experience/bench/measurements.jsonl` through `scripts/benchrec` — the one schema
  writer — so occupancy and kernel %bound share measurements.jsonl.

Gates (tiny, exact): `test_peak_equals_static_plus_transient_exactly` (invariant to the
byte; budget rows excluded from the held total) and a mutant control
`test_transient_zero_on_tiny_and_red_under_a_dropped_static_row`: drop the kv_pool static
row while holding the same peak and transient absorbs it; the gate pins transient below
a bound derived from the tiny activation shapes (`f32 [1,hidden]`), which the absorbed
16,384-byte pool row crosses — so the missing row cannot hide. Plus a benchrec
round-trip that asserts `shape.static + shape.transient == value`.

## Rule

A memory ledger that lists only named rows invites an unnamed bucket to hide bytes;
print the closure `peak = Σ static + transient`, share one renderer between the offline
table and the live /health endpoint, and gate it with a mutant that drops a static row —
transient growing past a shape-derived scratch bound is the red.
