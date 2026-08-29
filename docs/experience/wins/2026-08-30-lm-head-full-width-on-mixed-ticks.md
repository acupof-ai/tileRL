# lm_head ran over every position of every row on a mixed tick — cuda(H20), 2026-08-30

> Status: fixed (68eb137). Found by attributing memory, not by reading code.

## Context

Decode throughput was still climbing at B=32 (669.9 tok/s, +49% over B=16)
but B=64 OOM'd on a 95 GiB card. The pool arithmetic said that should fit:

| | |
|---|---:|
| weights (NVFP4) | 23 GiB |
| KV pool, 7264 blocks x 1 MiB | 7.3 GiB |
| GDN state, 64 slots x 144 MiB | 9.2 GiB |
| **total** | **40 GiB** |

55 GiB unaccounted. A first guess blamed a leftover process — there WAS one
holding 88 GiB, and clearing it did not fix the OOM, so that was a real but
separate problem.

## Root Cause

`scripts/probe_serve_mem.py` reports live tensors by shape after a few ticks.
At B=32 the list opens with:

```
8.53 GiB  x3   (3072, 248320) float32
2.84 GiB  x1   (6, 512, 248320) float32
```

3.05 GiB per copy, several live. That is **logits for 6 rows x 512 positions
x 248320 vocab**, when five of those six rows needed exactly one position.

`Model.forward`'s `last_only` was a bool, so it could only say "every row ends
at the same position":

```python
last_only = not chains and min(seq_q) == width
```

A mixed tick has `seq_q = [1, 1, 1, 1, 1, 512]` — decode rows end at 1, the
prefill row spans the bucketed width. `min(seq_q) != width`, the flag falls to
False, and lm_head — a `[248320, 5120]` projection, the largest in the model —
runs over all rows at full width. The cost scales with batch, which is why it
only became fatal as B grew.

## Fix

`last_only` takes the per-row valid lengths and gathers row `i` at
`seq_q[i]-1` before the final norm. Downstream indexing needed no change: the
prefill read was already `min(chunk, logits.shape[1]) - 1`, which becomes 0
once the row is narrowed, and decode rows already read position 0. Serving
only — training wants every position's logits and never passes a list.

## Rule

**A flag that collapses a per-row quantity into one bool will silently take
the expensive branch the moment the rows stop agreeing.** The bug was not a
wrong number anywhere; every line was locally correct, and `min(seq_q) ==
width` is a true statement about a uniform tick. It only became a 3 GiB
allocation because mixed ticks are the common case at high batch.

Second: **attribute memory before theorising about it.** The pool arithmetic
was right, the leftover process was real, and neither was the cause. One run
of a probe that lists live tensors by shape named the culprit in one line.
