# silu_mul's strided copy is 0.057% of a single-stream tick — REJECTED, and the note was wrong, 2026-09-03

> Status: **rejected at B=1.** The standing note said the fused `gate_up` layout costs
> "476 MiB of copies per MLP layer". Measured: that figure holds **only at 3584 rows**
> (7×512 prefill). On the single-stream decode path it is **0.5 MiB per layer, 34 MiB per
> tick, 0.037 ms of a 65 ms tick — 0.057%**. And at **1 row the copy does not happen at all**:
> the slices are contiguous. No change made.

## What the note claimed and what is true

`gate_up` writes `[M, 2I]`, so `gate = gu[..., :I]` and `up = gu[..., I:]` are strided views
and `backend.silu_mul` (`backend.py:1021-1022`) calls `_c()` on each, materialising both.
`I = 17408`, f32.

Measured with `scripts/probe_silu_copy_bytes.py` on the real 27B shape:

| rows | gate contiguous | up contiguous | `gu` | copied |
|---:|---|---|---:|---:|
| **1** | **True** | **True** | 0.1 MiB | **0.0 MiB** |
| 2 | False | False | 0.3 MiB | 0.3 MiB |
| **4** | False | False | 0.5 MiB | **0.5 MiB** |
| 8 | False | False | 1.1 MiB | 1.1 MiB |
| 32 | False | False | 4.2 MiB | 4.2 MiB |
| 512 | False | False | 68.0 MiB | 68.0 MiB |
| **3584** | False | False | 476.0 MiB | **476.0 MiB** |

Two corrections to the note:

1. **476 MiB is the 3584-row prefill number, not a per-layer constant.** The copy scales with
   rows, so quoting it without the shape overstates the decode path by **952×**.
2. **At one row there is no copy.** A `[1, 2I]` tensor's halves are each contiguous, so `_c()`
   is a no-op. Dense B=1 decode never pays this.

## The single-stream number

B=1 depth 3 submits **4 rows** (`LADDER_WIDTHS = (1, 2, 4, 8, 32)`, `spec.py:47`), so:

```
0.5 MiB/layer x 64 layers = 34 MiB per tick
34 MiB at 900 GB/s        = 0.037 ms
of the measured 65 ms tick = 0.057%
```

**Denominator note:** I first divided by a roofline estimate of MLP weight traffic
(8160 MiB/tick) and got 0.42%. That estimate implies a 9.1 ms tick against a **measured
62-70 ms** — a 7× contradiction, so the estimate is wrong (it counts fp4 MLP weights only,
while the tick carries every weight, the KV reads and the activations) and I did not report
the share built on it. The number above uses the **measured** tick instead, which is the only
denominator not derived from my own arithmetic.

## Why the fix is still cheap, and still not worth doing

The fix is real and small: have the fused GEMV write `[2, M, I]` so each half is its own
contiguous plane (verified in the probe at rows 1, 8 and 3584). The weight permutation is
load-time, so runtime cost is zero, and the change is MLP-local.

But it buys **0.037 ms of 65 ms** on the path that matters now. The noise floor is 1.16%,
so the win is **20× below what the harness can even see**. It touches the GEMV's output
contract, which the sm70 chunked path (`_sm70_chunks`, `backend.py:491`) writes into slice by
slice — a real chance of a slicing bug that only shows up at `M` not dividing 32, in a branch
with no CPU twin.

**Where it would pay: prefill.** At 3584 rows the copy is 476 MiB, and it is one of the three
allocations recorded as OOMing at exactly that shape
([`errors/2026-09-03-expandable-segments-is-load-bearing.md`](../errors/2026-09-03-expandable-segments-is-load-bearing.md)).
So this stays open as a **capacity** lever for wide ticks, not a throughput one — and the
entry it came from already said the padding, not any buffer's dtype, is the mechanism.

## Rule

**A byte count without its shape is not a measurement.** "476 MiB per MLP layer" reads like a
property of the layer; it is a property of one row count that appears only in prefill. The same
sentence was 952× wrong about the path I was about to spend time on, and it would have sent me
to rewrite a kernel's output contract for 0.037 ms.

Second: **when a share disagrees with a measured end-to-end number, the denominator is the
suspect.** My roofline said 9.1 ms where the card says 65. Rather than reconcile it, I replaced
it with the measured tick — a share against a real number needs no model of what else the tick
does.

## Gate

203 passed, 4 skipped, ruff clean. No GPU used: the copy is a layout property, measured from
tensor strides at the real `intermediate_size` (17408, read from the config, not assumed). The
probe asserts the `[2, M, I]` alternative is contiguous at every row count it checks.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | Mac | cpu | qwen38-27b cfg | copy at **1 row** | **0 MiB** (halves contiguous) |
| 2026-09-03 | (this) | Mac | cpu | qwen38-27b cfg | copy at **4 rows** (B=1 depth 3) | **0.5 MiB/layer, 34 MiB/tick** |
| 2026-09-03 | (this) | Mac | cpu | qwen38-27b cfg | copy at 3584 rows (7×512 prefill) | 476 MiB — **the note's figure** |
| 2026-09-03 | (this) | Mac | — | — | overstatement for the decode path | **952×** |
| 2026-09-03 | (this) | — | — | — | 34 MiB at 900 GB/s | **0.037 ms** |
| 2026-09-03 | (this) | — | — | — | **share of the measured 65 ms tick** | **0.057%** |
| 2026-09-03 | (this) | — | — | — | harness noise floor | 1.16% (**20× larger**) |
| 2026-09-03 | (this) | — | — | — | verdict at B=1 | **REJECTED** |
| 2026-09-03 | (this) | — | — | — | rejected roofline denominator | 8160 MiB/tick → 9.1 ms vs 65 measured |
