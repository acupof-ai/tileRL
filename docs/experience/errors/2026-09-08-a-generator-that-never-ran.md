# The MATH generator asked for a config that does not exist — 2026-09-08

**Status:** fixed. `scripts/math_jsonl.py` had never run; the `level` column it was fixed
to write on 09-05 was never exercised.

## Context

`errors/2026-09-05-the-eval-file-was-not-the-level-it-was-named.md` fixed a mislabeled eval
file: `math_test500.jsonl` was levels 3-5 mixed under a level-5 name, so its 80.0% base read
as a level-5 score against a 45.8% expectation from a 24-problem hand sample. Its fix was to
make `scripts/math_jsonl.py` write `level` into every row and print the histogram of what it
actually wrote. That entry names the remaining work: *"A level-5-only eval is an eval-only run
afterwards; the before-arm cache (#134) makes it cheap."*

Preparing that run is what found this.

## Root cause

`scripts/math_jsonl.py:45` read:

```python
ds = load_dataset("EleutherAI/hendrycks_math", "all", split=args.split)
```

**There is no `all` config.** The dataset has seven, one per subject:

```
BuilderConfig 'all' not found. Available: ['algebra', 'counting_and_probability',
'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
```

Confirmed against the repo's file list — `algebra/test-*.parquet` and six siblings, no `all/`
directory. The call raises before a single row is read, so **the script has never produced a
file**. Run 2's four JSONLs came from the throwaway builder the 09-05 entry says it deleted,
which is exactly why the eval file was missing its `--level` filter: the two builders were
different code, and only one of them was ever fixed.

**So the 09-05 fix landed on a script that could not run.** The `level` column, the
written-levels histogram, and the test asserting the boxing instruction all describe a code
path no data has passed through. The fix is real and correct — it just could not have been
exercised, and nothing said so.

Why nothing caught it: `datasets` is not a declared dependency (it is absent from
`pyproject.toml`), so it is not in `.venv` and no test imports it. `tests/test_recipes.py`'s
existing check reads the script as **text** — `assert '+ _INSTRUCTION' in text` — which is the
right shape for a hermetic test of a script that needs network, and it cannot see whether the
line above it executes.

## The premise that sent the work to the wrong machine

The 09-05 entry explains the throwaway builder this way: *"the pod has neither `datasets` nor
network, so `scripts/math_jsonl.py` could not run there."* **Both clauses are false.** Measured
2026-09-08: the pod has `datasets` 5.0.0 installed, and `HF_ENDPOINT=https://hf-mirror.com`
reaches the hub (only `huggingface.co` direct fails, with `Errno 99`). The pod generated the
level-5 file itself, in this session.

So the reasoning chain was: a false premise → build the data on the Mac with a throwaway script →
that script lacks the `--level` filter → the eval file is mislabeled. **The false premise is
upstream of the defect that entry documents**, which is why it is recorded here rather than left
as a footnote. `scripts/pod_sync.sh`'s "GitHub is unreachable from the pod" is true and is the
likely source of the generalisation; GitHub being unreachable does not make the pod offline.

The scope of a negative result cannot be widened past the one thing it was observed on.

## Fix

Concatenate the configs, enumerated from the repo rather than hardcoded, so a subject added
upstream is included instead of silently dropped:

```python
ds = [r for c in get_dataset_config_names(_REPO)
      for r in load_dataset(_REPO, c, split=args.split)]
```

Verified against source, both splits:

| split | rows | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|---:|
| test | 5000 | 437 | 894 | 1131 | 1214 | **1324** |
| train | 7500 | 564 | 1348 | 1592 | 1690 | **2304** |

`train` also carries 2 rows labelled `Level ?`, which the `--level` filter excludes. The
**2304 matches the recipe's claim** for `math_tr_5.jsonl` exactly, so the training file the
09-05 entry verified against the parquet shards is confirmed a second way. No level-5 row in
either split lacks a `\boxed{}` (0 dropped of 1324 and of 2304).

The level-5 eval file is generated and **checked against the source rather than against its own
column**: all 500 rows join to the source by problem text, the source's own `level` reads
`Level 5` for all 500, there are 500 distinct problems, and all 500 carry the boxing
instruction. The file's own `level` column is the script's claim; the join is the authority.
That check is the one the 09-05 defect lacked.

## Verification

**Run on both machines, and the two files are byte-identical**: `e5261691418bbf7d` from this Mac
(`datasets` 4.5.0, direct) and from the pod (`datasets` 5.0.0, `HF_ENDPOINT=https://hf-mirror.com`),
500 rows and `levels written: {'Level 5': 500}` each. So the fix is exercised on a real dataset in
two environments, not only against the stub below — which matters here more than usual, because
the defect was precisely a fix that had never executed.

`tests/test_recipes.py::test_math_jsonl_reads_every_subject_and_writes_the_level_it_filtered`
runs the generator as a subprocess against a **stubbed `datasets` module that raises on an
unknown config, like the real one**. Hermetic, because CI has neither `datasets` nor network —
and the stub's raising behaviour is what makes a revert fail here instead of only on a machine
that has the dataset.

| mutant | test |
|---|---|
| back to `load_dataset(_REPO, "all", ...)` | **FAILS** (`BuilderConfig 'all' not found`) |
| read only the first config | **FAILS** (`['4'] != ['4', '9']`) |
| control | passes |

The second mutant matters separately: a loop that stops at one config would produce a file of
the right shape with a sixth of the data, which no shape check catches.

## Rule

**A stub that cannot reject is not a test of a call that can be rejected.** The existing check
read the script as text and asserted a line was present; presence is not executability. When a
script's real dependency is unavailable in CI, stub it with the failure modes the real one has
— an unknown key raises — so a wrong argument fails the test rather than waiting for a machine
that has the dependency.

**And a fix to a script nobody runs is unexercised by construction.** Before fixing a
generator, run it once. `scripts/math_jsonl.py` carried a documented fix, a test, and a
CHANGELOG line for three days while being unable to read its first row.
