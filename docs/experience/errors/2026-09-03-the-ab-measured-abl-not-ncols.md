# The A/B measured `abl`, not `ncols` — a positional argument landed on the ablation switch, V100 sm70, 2026-09-03

## Context

`ncols=2` was accepted on the microbench (1.82× at M=32) and wired into the sm70
fp4 dispatch for its full-model gate. The dispatch called

```python
self._kernel("linear_fp4_gemv_sm70_m", Mk, 4, True, sh, 0, nc)
```

against a factory whose signature was

```python
def make_linear_fp4_gemv_sm70_m(target, M=8, GROUP=4, xh=False, sh=False,
                                min_blocks=0, abl=0, ncols=1)
```

Counting past `target`: `Mk`→M, `4`→GROUP, `True`→xh, `sh`→sh, `0`→min_blocks,
**`nc`→`abl`**. `ncols` was never set. Every prefill launch in both arms ran an
**ablation kernel — the ones whose own docstring says they return wrong numbers**.

## What it produced

| ctx | `TILERL_NCOLS=2` → `abl=2` (NO_SCALE) | `TILERL_NCOLS=1` → `abl=1` (X_REUSE) |
|---:|---:|---:|
| 512 | 8.47 | 1.48 |
| 2048 | 8.84 | 1.84 |
| 4096 | **9.35** | **2.35** |

Both readings are consistent with the ablations they actually ran, which is why
neither looked broken:

- `abl=1` X_REUSE makes every row read row 0's X, so the address is loop-invariant
  and **ptxas deletes 85% of the loads** (LDG 363 → 53). 8.79× on the microbench;
  3.8× here. It reads as a spectacular prefill win.
- `abl=2` NO_SCALE drops the widen+scale, measured **0.92× — nearly free**. So the
  "ncols=2 arm" sat right on the recorded 8.92 baseline and looked like a null result.

So the table read as "ncols=2 is a 4× regression against ncols=1", the exact
opposite of the microbench, and both numbers had an innocent explanation.

## What caught it

Not the direction reversal — a reversal invites a mechanism story. What caught it
was the **magnitude against a recorded number**: `TILERL_NCOLS=1` is by
construction the shipped code path, and the shipped path's prefill at 4096 is
**8.92 ms/token** in
[`wins/2026-09-02-prefill-chunk-loop-was-quadratic.md`](../wins/2026-09-02-prefill-chunk-loop-was-quadratic.md).
2.35 is 3.8× faster than the kernel it is supposed to *be*. A flag cannot make
the code it disables faster than the code itself.

The standing rule fired: *a measurement that contradicts a known end-to-end number
by more than 2× indicts the instrument.* Nothing was reported; the next step was
reading the call site, not explaining the number.

The parity signal was in the same output and I nearly missed it: **first-token id
was 0 in every arm at every context**. Greedy argmax over a 27B's real logits does
not return id 0 three times.

Except it does, here — the corrected run read 0 as well, on kernels whose text is
provably right. The prompts are synthetic (`range(base, base+ctx)`), so the ids were
never a check at all: they read the same for a correct and an incorrect kernel. That
is worse than a missed tell, and the fix is `scripts/parity_ncols.py`, which decodes
real greedy continuations at M=1 and M>8 and compares them between arms.

## Root cause

A factory with five tuning parameters and three *measurement* switches accepted all
eight positionally, so the serving path could reach a "returns deliberately wrong
numbers" flag by miscounting. The ablations were built for a harness that always
named them (`abl=a`); the dispatch was the first caller to count.

## Fix

Three lines, at the depth where the class of bug dies rather than this instance:

1. `min_blocks`, `abl`, `ncols` are **keyword-only** (`*,` in the signature). A
   positional 6th argument is now a `TypeError`, not a wrong kernel.
2. `Backend._kernel(name, *args, **kw)` forwards keywords and keys its cache on
   them, so the dispatch can name a late parameter without counting.
3. The dispatch passes `ncols=nc`.

`tests/test_ncols_contract.py` grew two assertions — the signature kinds, and
`"ncols=nc" in` the dispatch source — and both negative controls were verified to
fail when reverted.

## Rule

**A flag that makes code deliberately wrong must not be reachable by position.**
Measurement switches and tuning parameters share one signature here; the tuning
half is called from serving, the measurement half returns garbage by design. Keyword-only
is the whole guard, and it is free.

Second: **the A/B arm that is supposed to be "current behavior" is a control — price
it against its recorded number before reading the delta.** The comparison between
the two arms was internally consistent and entirely fictional. One arm had a
published value; checking it is what ended this.

Third: **a bench's correctness column is worth exactly what its input is.** The
first-token ids looked like a free parity check sitting in the timing output, and
they were noise — synthetic prompts, id 0 either way. A wrong-number bug does leave
tells, but only in an output that could have differed; here that took a second
script decoding real text.
