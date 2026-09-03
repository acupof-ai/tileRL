# The draft readout was reduced on two paths of three — V100, 2026-09-04

> Status: fixed. Task #43 cut the draft's vocab readout to the row it reads, and
> `DraftHead.step` — the path every tick actually goes through — was not one of the
> call sites changed. 485 MiB per prefill chunk, invisible to every correctness gate.

## Context

lc17 (task #57, the decode-rate curve at 8K–32K) died in warmup at ctx=8192 with a
4146-block pool sized for 32768:

```
torch.OutOfMemoryError: Tried to allocate 486.00 MiB. 466.38 MiB is free.
  backend.py:534 in linear_fp4          <- y2 = torch.empty(M, N, float32)
  spec.py:323 in DraftHead.forward      <- self.trunk._linear(backend, x, head)
  spec.py:404 in DraftHead.step
  engine.py:793 in _run_forward
```

## Root Cause

`M=512, N=248320, f32` is **485.9 MiB** — the allocation, to three digits. That is
the trunk's vocab readout run over a whole prefill chunk by the *draft*, which then
reads one row of it per request.

Three call sites reach `forward` with a multi-position tick, and `last_only` was
passed by two:

| caller | line | last_only |
|---|---|---|
| decode graph capture | `engine.py:239` | yes |
| trunk verify tick | `engine.py:757` | yes (`seq_q`) |
| **`DraftHead.step`** | `spec.py:404` | **no — defaulted to False** |

Task #43 reduced the readout and measured -1.53 GiB at B=8 ctx=512. That number came
off the graph path. The reduction machinery inside `forward` (`spec.py:316`) was
correct and gated; the eager `step` never asked for it.

**Why no gate saw it.** The unreduced path returns the *same token* — `step` sliced
`logits[rng, last]` itself, so output is byte-identical either way. The only
observable is peak memory, and every spec test runs the tiny model where the readout
is 6 x 300 floats. The existing parity test's spy even *read* the wide tensor
(`out[i, -1]`) and its comment says the index "is the last valid row either way" —
the wide shape was documented as normal rather than flagged.

## Fix

```python
logits = self.forward(..., hidden_out=dh, last_only=sq)
tok, prob = backend.greedy(logits)          # was logits[rng, last].unsqueeze(1)
```

`sq` is the per-row valid width `step` already builds, which is exactly what
`engine.py:757` passes for the trunk. `hidden_out` is appended at full width inside
`forward` (before the reduction), so `dh[-1][rng, last]` is unchanged — the chain's
own hidden still comes from the right position.

Per prefill chunk at the shipped `--max-num-batched-tokens 512`: **485.9 MiB → 0.95
MiB**, 512x. Decode ticks are already one position wide and are unaffected.

## Gate

The parity spy asserts the shape rather than reading around it:

```python
assert out.shape[1] == 1 or np.asarray(ids).shape[1] == 1, (
    f"draft readout is {out.shape[1]} positions wide for a "
    f"{np.asarray(ids).shape[1]}-position tick: pass last_only")
```

Negative control: reverting the one-line fix (with `__pycache__` cleared, per
`memory/pyc-cache-defeats-negative-controls.md`) fails **all six** arms of
`test_engine_draft_matches_full_context_draft` at `7 positions wide for a
7-position tick`, and nothing else in the suite. 248 passed with the fix.

The gate lives in the *existing* parity test because that test already drives the
real `step` across chunked prefill, ragged widths and block edges — the four cases
that produce a wide tick. A new test would have had to rebuild all of that to check
one shape.

## Rule

**When a fix is "reduce X at the call site", the unit of work is every call site, and
the count belongs in the entry.** #43's entry names a saving and a mechanism; it does
not say how many callers were changed, so nothing carried the fact that a third
existed. A shape assertion at the *callee* would have covered all three at once —
which is where this gate now sits, one frame in from where the bug was.

Second: **a defect whose only observable is peak memory has no correctness gate by
construction.** This one survived #43, #42, #44 and #45 — four consecutive
memory-shaped investigations on the same code — because each of them measured a
configuration whose readout happened to fit. It surfaced when the pool grew, not when
the code changed.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-04 | 4294c50 | V100 | cuda sm70 | 27B, draft, ctx 8192 | lc17 warmup | **OOM, 466 MiB free** |
| 2026-09-04 | (this) | — | — | — | draft readout per 512-token chunk, before | 485.9 MiB |
| 2026-09-04 | (this) | — | — | — | same, after | **0.95 MiB (512x)** |

Source: `$HOME/tilerl-logs/lc17.log:181-212`.

## Still open

Task #57 is still unmeasured — the curve this found the bug on has produced no rows.
Relaunch after the fix reaches the pod.
