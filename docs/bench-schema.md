# Bench schema — one measurement is one record

The store is `docs/experience/bench/measurements.jsonl` (append-only, one JSON
object per line). Collectors append with `scripts/benchrec.py`; every view
reads that file and nothing else. A record missing a required field is
rejected at write time, not warned about later.

## Why

Five failures in one day, all the same shape — the population was never
written down: a build (eager 14.7 vs fused+graph 94.6, 6.4x), a length cap
(train 2048 vs eval 6144), a compile count (a 250 s cold run with 12
compiles), a sha, a type. A point estimate without its population answers a
different question every time someone reads it.

## Required fields

| Field | Type | Rule |
|---|---|---|
| `metric` | str | must exist in `docs/bench-metrics.json` |
| `value` | number | the measurement |
| `unit` | str | must equal the registry's unit for the metric |
| `target` | str | `cpu` / `metal` / `sm90` / `sm70` |
| `build` | str | `eager` / `fused` / `fused+graph` / `fused+graph+draft` |
| `model` | str | config name, e.g. `27B-nvfp4`, `tiny` |
| `shape` | object | non-empty; must contain the metric's registry-declared keys (decode → `batch`,`ctx`; eval → `cap`; reuse → `turn`) |
| `warm` | object | `state` ∈ `cold`,`warm`; `compiles` int ≥ 0 — **absent is rejected** (0 is an assertion, absent is unmeasured) |
| `n` | int | ≥ 1, the number of timed windows |
| `spread` | number | relative dispersion (sd/mean or (max−min)/median); 0.0 when `n=1` |
| `device` | object | `name` (GPU model); `card` int, required on sm90/sm70 |
| `sha` | str | git sha of the code under test (pod: stamped from `.synced_commit`) |
| `cmd` | str | the exact command that produced the row |
| `floor` | object | `value` > 0, `unit` (must equal the record's unit — a floor in other units is a forged floor), `kind` ∈ `bandwidth`/`compute`/`roofline`/`measured-best`/`baseline`, `derivation` non-empty |

`floor.derivation` must state the computation with numbers. A `baseline`
floor (random-guess 25%, no-reuse 1.0x) must name the null it is measured
against — "no known floor" is rejected.

## Reruns

A rerun of the same population appends a new row with `supersedes` (the old
row's id) and `note` (why the rerun). Both rows stay in the file; views take
the newest non-superseded row per population key. A rerun is visible — that
is the point of append-only.

## Derived, never stored

`id` (content hash), `date`, `direction`, `weight`, `gap`. Weights live in
`docs/bench-metrics.json` with their derivation; **changing a weight is a
separate commit with the derivation in its body** — it reorders
`--questions`, which is a default flip.

`gap = floor/value` for higher-is-better metrics, `value/floor` for
lower-is-better. The question list sorts by `gap × weight`.

## Views

- `tilerl bench --table` — four-target matrix; an empty cell says so, never a silent skip
- `tilerl bench --readme` — the generated README rows (reuse, SSD restart)
- `tilerl bench --regress` — newest vs previous per key; rows with `n=1` are excluded (no dispersion, no regression claim)
- `tilerl bench --questions` — all current rows by `gap × weight` desc

## Selftest

`python3 scripts/benchrec.py` — a good world accepts; three bad worlds
(missing field, `warm` without `compiles`, forged floor) reject; a legal
`n=1` row accepts but is fenced out of the regression view. The system proves
it goes red.
