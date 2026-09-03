# serve --blocks 0 --draft OOMed at its own default slots — V100, 2026-09-04

> Status: fixed by ordering. The KV fit spent memory the draft head's weights
> still needed, and the failure landed three frames later inside a twiddle.
> `--blocks 0` is `serve`'s default, so this was the shipped path.

## Context

Task #62 asked for one thing: `_fit_blocks`'s CUDA branch returns early for
non-CUDA and every bench passes `num_blocks` explicitly, so `serve --blocks 0` had
never been exercised end to end since `b36e45a` corrected its draft term. One run
that prints the fitted pool and answers one request, at serve's own defaults.

## Root Cause

At `--slots 16 --max-batch 8 --depth 3` it does not print a pool. It dies:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 170.00 MiB.
GPU 0 has a total capacity of 31.74 GiB of which 104.38 MiB is free.
  reference.py:245 in _twiddle_fp4
  backend.py:711 in materialize
  engine.py:367 in Engine.__init__
```

The allocation order in `build_engine` is the whole cause:

| step | line | takes |
|---|---|---|
| trunk `materialize` + `empty_cache` | 1166-1174 | reclaims the load reserve |
| GDN state pool | 1179 | slots x width |
| **`_fit_blocks`** | 1192 | **2/3 of what is free** |
| `PagedKvPool` | 1194 | that |
| `PrefixStore` | 1214 | a quarter of the rest |
| **`Engine.__init__` quantizes the draft's weights** | 1215 → 367 | **whatever is left** |

The draft's own fp4 weights were charged to nothing. `_fit_blocks` already takes a
`draft_layers` argument, which reserves the draft's KV **mirror** — a different
quantity from its **weights**, and the name made it easy to believe both were
covered.

**Why it had not shown before.** The recorded 4046-block measurement
(`wins/2026-09-03-allocator-reserve-was-the-context-ceiling.md`) ran
`probe_kv_ceiling.py` at `--slots 3`, which leaves 3.04 GiB free after everything —
enough for the draft by accident. `serve` defaults to `--slots 16`, where the state
pool is several times larger (`step_states` scales `slots x width`, not
`max_batch`), so 2/3 of a much smaller remainder leaves nothing. A default that
differs from the config the number was measured at is how an unexercised path stays
unexercised.

## Fix

Serve the draft's weights in `build_engine`, **before** anything reads free memory —
the same ordering the state pool already uses, and for the same reason: allocate,
then *measure* what is left instead of estimating it. `Engine.__init__` keeps its
call for a caller who constructs an `Engine` directly with an unquantized draft;
`_quantize_draft` returns its input once packed, so that path costs a dict copy.

Measured after the fix, same command, serve's defaults:

| | slots=3 | slots=16 |
|---|---:|---:|
| before | 3927 blocks = 62832 tok | **OOM in twiddle, 104 MiB free** |
| after | 3927 blocks = 62832 tok | **757 blocks = 12112 tok, answers** |

Both answer, and with byte-identical output (`[10248, 61354, 62290, 44576, 92, 93,
198, 10]`), so the pool size does not move the tokens. The 5.19x smaller pool at
5.3x the slots is the state pool's real cost, now charged before the fit rather than
crashing after it.

**Task #62's own number, incidentally confirmed.** `_fit_blocks` at slots=3 returns
**3927 blocks = 62832 tokens** against a prediction of exactly 3927/62832 derived
from `b36e45a`'s corrected draft term (1/16 of a block per draft layer, not 1/32).
The 4046/64736 in `cli.py`'s `--blocks` help and the 64912 in `engine.py`'s reclaim
comment were both pre-fix; both now carry the measured figure.

## Gate

`test_fitting_the_kv_pool_happens_after_the_state_pool` gains a drafted arm
asserting `seen[:3] == ["draft", "state", "fit"]`. Negative control run — reverting
the move fails it at `index 0 diff: 'state' != 'draft'`, and nothing else in the
suite. 248 passed.

The gate also caught something in the fix itself: the spy recorded
`["draft", "state", "fit", "draft"]`, i.e. `Engine.__init__` re-serves. That is
correct (idempotent) but it is a second dict copy, and the assertion is written to
allow exactly one such re-serve rather than to hide it.

## Rule

**A default that no measurement ran at is an untested default.** The fit was
measured at `--slots 3` and shipped with `--slots 16`; the arithmetic was right and
the config was not. When a number goes into a help string, put the flags it was
measured at in the same string — `cli.py`'s now says `--slots 3`, which is what
would have made this visible without a run.

Second: **an ordering bug reports at the allocation that fails, never at the one
that overspent.** The traceback named `_twiddle_fp4` — a function with no idea a KV
pool exists. Both of this file's ordering constraints were found the same way (the
`empty_cache` one OOMed inside `PagedKvPool`), which is why the gate asserts a
*sequence* and not a byte count: the sequence is checkable off-GPU, and the bytes
are card-specific.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-04 | 2b28c10 | V100 | cuda sm70 | 27B, slots=16 | `--blocks 0 --draft`, before | **OOM, 104 MiB free** |
| 2026-09-04 | (this) | V100 | cuda sm70 | 27B, slots=16 | `--blocks 0 --draft`, after | **757 blk = 12112 tok, answers** |
| 2026-09-04 | (this) | V100 | cuda sm70 | 27B, slots=3 | fitted pool | 3927 blk = 62832 tok |
| 2026-09-03 | — | V100 | cuda sm70 | 27B, slots=3 | same, pre-`b36e45a` | 4046 blk = 64736 tok (withdrawn) |

Source: `$HOME/tilerl-logs/fit1.log` (the OOM), `fit2.log` (slots=3), `fit3.log`
(slots=16 after the fix).

## Still open

`--slots 16` at `--depth 3` costs 5.19x the KV pool on a 32 GB card. That is now
visible instead of fatal, but nothing has priced whether 16 slots is worth 12112
tokens of context against 3 slots at 62832 — the throughput side of that trade is
unmeasured (task #65).
