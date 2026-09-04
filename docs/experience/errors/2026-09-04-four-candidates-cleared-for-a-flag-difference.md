# Four code candidates cleared for a difference that was two flags — V100, 2026-09-04

> Status: closed, cause found, and the cause was the comparison rather than the
> code. Two runs of "the identical command" differed in `--tokens`, which IS the
> measurement window, and I spent four GPU rounds eliminating commits.

## Context

Task #57's long-context sweep read tok/forward **2.89 at ctx=8192** where the
09-03 entry recorded **2.03–2.10 at 1024–4096** on what I believed was the same
harness, same prompt, same everything. A 1.376x acceptance jump is large enough to
move `spec.py`'s `p=0.6536`, which feeds the whole Task #22 reject, so it had to be
attributed before anything downstream could be trusted.

## Root Cause

`lcfix2.log` (09-03) and `lc19.log` (today) print different pool sizes:

```
lcfix2:  pool: 554 blocks (1177 MiB) sized for ctx=4096, sweeping [1024, 2048, 4096]
lc19:    pool: 562 blocks (1194 MiB) sized for ctx=4096, sweeping [1024, 2048, 4096]
```

The pool formula is `b * ceil((max_ctx + tokens + 2*(1+depth)) / 16) + 32`, and at
b=2, ctx=4096, depth=3 it inverts uniquely:

| `--tokens` | blocks |
|---:|---:|
| 32 | 550 |
| **64** | **554** |
| 96 | 558 |
| **128** | **562** |

So 09-03 ran `--tokens 64` and today ran `--tokens 128`. **`--tokens` is the
measurement window**: `measure()` closes the window at the first completion, and
tok/forward is `tokens_generated / decode_forwards` *within it*. Acceptance is not
constant across a generation — on a random-vocabulary prompt the model leaves the
prompt's distribution and enters its own, which the draft predicts better — so a
64-token window and a 128-token window average different stretches and report
different numbers from identical code.

That also explains the shape that made me disbelieve the instrument: 1.202x /
0.990x / 1.143x at 1024/2048/4096, with the middle point flat. The window covers a
different fraction of the generation at each ctx (a longer prefill delays the first
completion), so a fixed `--tokens` difference lands differently per point. No
physical effect skips the middle point; a window artifact does.

## What it cost

Four candidates eliminated, each with its own instrument, none of them guilty:

| candidate | how eliminated | cost |
|---|---|---|
| `b9af605` draft block growth | arithmetic — every ctx is a multiple of `BLOCK_TOKENS=16`, so `len(blocks)*16 <= seq_len-1` never fires; its minimal repro is a 15-token prompt | free |
| `bd48e06` merge (`head_key`, `qwen36_27b`) | diff — inline ternary replaced by an identical property; the config changes are all in a model we do not run | free |
| `5a288e6` readout narrowing | **measured** — pre-fix `spec.py` reads 2.44 at ctx=1024, bit-identical to post-fix | one pod run |
| harness repeatability | measured — `--repeats 3` reads `s_tpf = 0.0%` at every point, twice, in two processes | two pod runs |

Three pod runs and four rounds of reasoning, against two log lines I could have
diffed first. Worse, I published a wrong intermediate verdict twice: first
"the cause is code" off the 1024 point alone, then "the harness has ±20%
repeatability" off three cross-run points, which the 0.0% spread refuted an hour
later.

## Fix

Two changes, both to the harness rather than the runtime.

**1. Print every parameter the numbers depend on**, not just the pool:

```
config: depth=3 tokens=128 batch=1 slots=3 max_batch=2 repeats=2
        prompt=randint(vocab,seed=1000+i)
```

The pool line was already there and I *did* read it — 554 against 562 — but it
requires inverting a formula to learn what differs. A parameter line is what makes
a mismatched comparison visible without arithmetic.

**2. `timed()` averages its draws and reports the spread.** It already took three
draws per point and returned the **last one**, discarding two: the repeatability
data was being generated and thrown away. Now the mean of `--repeats` draws plus
`(max-min)/mean` for both columns.

Measured spreads, which is the second finding here: **tok/forward is 0.0% —
exactly, at every context, in every run.** Greedy sampling with a fixed prompt
makes it a deterministic integer ratio. tok/s is 0.3–0.9%. So a single draw of
acceptance is *reproducible*, which is why the cross-run difference could not be
noise and had to be a configuration difference — the zero spread is what finally
pointed at the flags.

## Gate

`_self_check` fakes `measure` with three known draws and asserts the first is
absorbed untimed, the mean is of the rest, and the spread is `(max-min)/mean`.
Negative control: returning `tps[-1]` instead of the mean fails it at
`got 36.0`. Runs on the GPU-less box in one second; the failure it prevents needs a
20-minute pod job to appear and is invisible when it does.

## Rule

**A number is only comparable to another number measured at the same flags, and
"the identical command" is a claim about a command line I have to actually
compare.** Both runs came from a script I wrote, one from `lc19.sh` and one from
`lcfix2`'s invocation, and I never diffed them — I compared the numbers and
searched for a mechanism. The pool line even encoded the difference.

Second, and this is the more general one: **when an instrument's own spread is
zero, a difference cannot be noise, so the search must move to configuration
before it moves to code.** I had the 0.0% in hand and used it to conclude "the
difference is real, therefore it is code" — real and code are not the same
disjunct. Configuration is also real.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-04 | (this) | V100 | cuda sm70 | 27B d3 B=1 | `--tokens` inverted from 554 blocks | **64, uniquely** |
| 2026-09-04 | b20ca9b | V100 | cuda sm70 | 27B d3 B=1 | tok/fwd spread, 3 draws, ctx 1024/2048/4096 | **0.0% / 0.0% / 0.0%** |
| 2026-09-04 | b20ca9b | V100 | cuda sm70 | 27B d3 B=1 | tok/s spread, same | 0.8% / 0.3% / 0.6% |
| 2026-09-04 | 5a288e6^ | V100 | cuda sm70 | 27B d3 B=1 | tok/fwd at ctx=1024, pre-readout-fix | 2.44 (= post-fix) |

Source: `$HOME/tilerl-logs/lcfix2.log` (09-03, `--tokens 64`), `lc19.log`,
`lc20.log` (`--repeats 3`), `lc21.log` (pre-fix `spec.py`, md5 `7dc78b55`).

## Still open

The 09-03 entry's three tok/forward values are valid only at `--tokens 64` and do
not carry that label. They are not withdrawn — they were correctly measured — but
they cannot be compared to any 128-token row, and the entry says nothing about the
window. Task #57's curve therefore reports **tick cost only**; its acceptance
column mixes two windows and is not a curve.
