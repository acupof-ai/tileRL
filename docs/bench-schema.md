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

A hand-written `commit` is the same failure in the provenance field:
`wins/2026-09-03-batched-selector-walk.md:80` cites `40bc83c` for a B=1 row
that commit cannot produce (B=1 landed in #58) — the sha was main's nib at
entry-writing time, not the tree that produced the number. It lay for three
days, was cited by the README, and was handed out as a repro baseline. So the
collector takes the sha itself, in the tree it ran in, and `dirty` says
whether that sha fully names the tree.

A first measurement has no history: `--regress` is silent on both sides, so
the physical floor is its **only** alarm. That is why beating a hard floor is
`IMPLAUSIBLE` rather than a win — the too-good number is the most common shape
of bad measurement (a cost missed, the wrong population, a gate reading an
always-true field), and the first measurement is exactly when nobody is
watching.

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
| `commit` | str | full 40-hex git sha of the code under test, **self-collected** (`git rev-parse HEAD` in the tree that produced the number) — never hand-filled; validated to exist in the repo (pod: stamped from `.synced_commit`) |
| `dirty` | bool | the tree had uncommitted changes (`git status --porcelain` non-empty); a sha cannot fully identify a dirty tree (pod: `.synced_dirty`) |
| `cmd` | str | the exact command that produced the row |
| `floor` | object | `value` > 0, `unit` (must equal the record's unit — a floor in other units is a forged floor), `kind` ∈ `bandwidth`/`compute`/`roofline`/`measured-best`/`baseline`, `derivation` non-empty |

`floor.derivation` must state the computation with numbers. A `baseline`
floor (random-guess 25%, no-reuse 1.0x) must name the null it is measured
against — "no known floor" is rejected. The derivation is the one field the
machine cannot check, so review must: **every new `floor.derivation` is
recomputed by its reviewer** — the arithmetic, not just the prose
(`wins/2026-09-09-the-first-floor-derivation-failed-its-own-check.md`).

## Staleness

Records carry `date` (automatic), but no view rejects an old baseline:
`--regress` is advisory, and comparability lives in the population key — same
key means same thing measured, however far apart. **If `--regress` ever becomes
an automated gate, add `asserted_at` and make the gate reject stale facts** —
an automated gate reading an expired fact and deciding anyway is the aupai
failure mode; an advisory print is not it. A freshness cutoff before that would
hide long-horizon regressions, which are the view's purpose.

A flag that changes a metric's meaning is a population field, not prose: add
it to the metric's registry `shape` keys (spec `depth`, SSD `arm`) so the key
names the condition — `tied@lam=0.1` in the record, never `tied` with the
lambda in a comment. A value whose condition lives outside the key is not
falsifiable from the store.

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
lower-is-better. A gap's meaning depends on its floor kind: physical floors
(`bandwidth` / `compute` / `roofline` / `baseline`) make gap a **headroom**
number (how far the limit is), `measured-best` makes gap a **regression**
number (how far below our own best). The two never share a sorted column.

## Collector helpers

`benchrec.add_record_args(ap)` adds the population flags (`--build` /
`--target` / `--device-name` / `--card` / `--model-name`);
`benchrec.record_common(args, build=...)` builds the shared fields, demanding
`--build` when the script cannot see the build (every server client) and
`--card` on sm90/sm70. `benchrec.measured_best_floor(record, lower_is_better)`
is the floor for a metric with no computed roofline yet: the population's best
accepted value, or this measurement on first sight. A script that derives its
build from its own flags passes it to `record_common`; a script that cannot see
it leaves `--build` required.

## Views

Every view prints a coverage line (`N metrics declared, M measured`; `--table`
adds targets covered) — an empty store must make noise, not read as a clean
bill of health.

- `tilerl bench --table` — four-target matrix; an empty cell says so, never a silent skip; the gap column carries its floor kind (`roof`/`bw`/`compute`/`base` = headroom, `best` = vs our own best)
- `tilerl bench --readme` — the generated README rows (reuse, SSD restart); coverage as an HTML comment so the table stays paste-safe
- `tilerl bench --regress` — two sections: newest vs previous per key (`n>=2` only, no dispersion, no regression claim), and every `measured-best` row standing below its population's best (FAIL past 1.05x). The symmetric side: a new best that jumps more than 1.2x beyond the previous best is `IMPLAUSIBLE — explain or reject` (1.2x is ~10x the 1.7% run-to-run spread; a real jump gets the explain line it deserves)
- `tilerl bench --questions` — headroom against **physical floors only**, by `gap × weight` desc; above it, `no measurement at all` (declared in the registry, never measured) and below it `no physical floor — needs a derivation` (measured, no physical floor) — both sorted by weight, both louder than any ranked row. A value that **beats a hard physical floor** (bandwidth/compute/roofline, not baseline — the null is meant to be beaten) is `IMPLAUSIBLE — explain or reject`, printed before everything else: a too-good number is the most common shape of bad measurement (a cost missed, the wrong population, a gate reading an always-true field), and the only alarm a first measurement has

## Selftest

`python3 scripts/benchrec.py` — a good world accepts; bad worlds reject
(missing field, `warm` without `compiles`, forged floor, missing `commit`,
a `commit` not in this repo, missing `dirty`); a legal `n=1` row accepts but
is fenced out of the regression view. The system proves it goes red.
