# scripts/ is 44% of the tree and needs no changes

**Date:** 2026-09-08 · **Verdict:** closed, zero changes · **Head:** `b4e9360`

## Context

`scripts/` holds 200 Python files and 28,890 lines — 44% of the repository. It reads as
the obvious cleanup target: no package, no imports from `src/`, filenames that look like
one-off probes (`_sweep_gemv3.py`, `probe_gdn_assertion1.py`), and 85 copies of the same
`sys.path.insert` bootstrap. Three separate readings proposed deleting part of it or
consolidating the duplication.

The audit enumerated instead of searching, and the answer is that nothing should change.

## What the enumeration found

Six entry-point sets were built and each file classified by set membership, so a file is
unreached only when it is absent from all six *enumerated* sets. A keyword search would
only have proven what someone thought to search for.

| bucket | files | lines |
|---|---:|---:|
| LIVE — reached by pyproject, CI, an import, a shell invocation, or a non-experience doc | 82 | 14,039 |
| PROVENANCE — cited only by `docs/experience/**` | 84 | 10,749 |
| UNREACHED — absent from all six sets | 34 | 4,102 |

**PROVENANCE is larger than LIVE.** `scripts/` is the repository's biggest store of
documented evidence: 84 files exist because a wins or errors entry quotes a number they
produced, and deleting one destroys the provenance of a figure still in the tree.

All 34 UNREACHED files were opened by hand. Every one is load-bearing, in one of two ways.

Some carry an argument no entry holds. `probe_ssd_arrival_rate.py` (389 lines, the
largest) measures the arrival side of the SSD cap: the drain rate is known (240 MiB/s, a
full `max_pending=32` queue in 42.8 s) and `ssd_offered` is a count, so nothing says
offers/second. Its docstring also records a rejected first version — max(offers per
poll)/interval read 7.95/s at `--interval 0.25` and 19.50/s at 0.05, both exactly
(1 or 2)/interval on a workload whose mean interarrival is 1.07 s, an instrument artifact.
Deleting the file deletes the only record that the obvious instrument is wrong.

Others produced a number an entry quotes without naming the file.
`probe_norm_cast_once.py` (153 lines) is behind the 2.0314 ratio in
`wins/2026-09-03-qk-norm-casts-once.md:28` — the entry states the ratio and never the
script. That link is semantic and no mechanical set can find it, which is exactly why
these 34 had to be read rather than classified.

**Deletable files: 0.**

Re-running `scripts/audit_scripts_entrypoints.py` after this entry lands reports 87/32
rather than 84/34: naming `probe_ssd_arrival_rate.py`, `probe_norm_cast_once.py` and
`step_phase_split.py` above moved them into PROVENANCE, and the enumerator itself reads
UNREACHED on its first run. The table is the classification at `b4e9360`, before this file
existed. The audit is inside its own input set.

## The consolidation is 0.16% and not worth its diff

Three duplicated blocks were measured against the whole tree.

**`sys.path.insert` — 85 files, and it must stay inline.** It is bootstrap, not
duplication: a shared preamble that 100 scripts import must itself be importable before
`sys.path` is fixed. The alternative — depend on `uv run` — breaks the 53 files whose
only documented invocation is a bare `python3 scripts/…`, which is the pod pattern
(`pod_run.sh:82,87` does `cd $REMOTE_DIR` then exports `PYTHONPATH=$REMOTE_DIR/src:…`).

**The two post-bootstrap lines absorb, and buy almost nothing.**

```
files touched            : 121
lines removed            : 168   (TILERL_TARGET setdefault 49 + get_backend import 119)
lines added (1 import ea): 121
net                      : -47 of 28,890 = 0.16%, plus ~6 for the preamble itself
```

Net −41 lines across 121 files. The line count is not the real cost: those 121 files are
the provenance of measurements still quoted, so a mechanical commit lands in every
probe's `git log` ahead of the change someone is actually looking for.

**argparse is the biggest block and the one that must not be touched** — 886 lines across
123 files. It cannot be shared, because the names are not the same parameter:

| flag | files | distinct signatures |
|---|---:|---:|
| `--out` | 24 | 11 (`""`, `default=""`, `/tmp/pool.jsonl`, `/work/pad_histogram.json`, …) |
| `--layers` | 27 | 8 (`"3,4"`, `type=int default=2 / 4 / 48 / 64`) |
| `--gen` | 14 | 12 |
| `--iters` | 16 | 5 (`default` 5 / 10 / 20 / 30 / 50) |

186 distinct flags, **106 appearing in exactly one file**. Only 11 appear in ≥10 files and
those 11 disagree on defaults and types. A common parser would need every call site to
override, adding back more than it removes, and would leave a trap: a flag that looks
shared and is not.

## Rule

**A link is evidence in proportion to how often it fails to appear.** Four instances of
the same defect turned up while building this audit, three of them in the instrument
rather than the system:

- every script's docstring says `Run: scripts/<self>.py`, so the self-edge made 113 of 200
  files look imported;
- a proposed "linked by a commit that touches both" set matched 191 of 200, because the
  median linking commit touches 130 files;
- `--out` in 24 files is a name collision, not a shared parameter;
- and the inverse — a semantic link that appears almost nowhere mechanically is exactly
  the one worth following, which is why the 34 UNREACHED files had to be read by hand.

An edge that fires almost everywhere cannot discriminate, and neither classification nor
deletion may rest on one.

Second rule, for the shape of the target rather than the instrument: **the part that looks
most deletable is the part that is load-bearing.** The filenames that read as abandoned
one-offs are the provenance store; the duplication that reads as sloppy is bootstrap that
cannot be shared; the largest duplicated block is the least shareable.

## Left undone, deliberately

19 scripts resolve `src` against the **cwd** (`sys.path.insert(0, "src")`), so they run
only from the repository root, and nothing says so. Neither known invocation point hits
it. Measured, four combinations, only one fails:

| interpreter | cwd | result |
|---|---|---|
| `uv run` | repo root | ok |
| `uv run` | `scripts/` | ok — `scripts/src` does not exist, the insert is inert, the editable install answers |
| bare `python3` | repo root | ok |
| bare `python3` | `scripts/` | `ImportError: No module named 'tilerl'` |

**What keeps the pod safe is a rule, not the environment.** `pod_run.sh:82,87` does
`cd $REMOTE_DIR` and then exports `PYTHONPATH=$REMOTE_DIR/src:…`, so cwd and `PYTHONPATH`'s
first entry name the same tree. That is two lines of one script. A launcher that does not
go through `pod_run.sh`, or a hand-typed `cd` elsewhere, reopens the precondition. Recorded
rather than fixed, because no requirement has appeared.

Two claims about safety here have different strengths and must not be read as one. *A
nonexistent directory on `sys.path` is inert* is structural — Python skips it, which is why
row 2 above passes. *No invocation runs one worktree's script under another's environment*
is not: it holds until the next call site. Silently shadowing the editable install is
constructible today (an insert path that exists and differs from the bound tree wins,
because it goes first, with no warning); nothing performs it. The first sentence protects
one row of the table, the second protects the conclusion, and only the first survives a new
caller.
