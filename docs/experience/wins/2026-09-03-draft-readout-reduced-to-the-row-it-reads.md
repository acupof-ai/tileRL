# The draft readout allocated 191× what it read — fixed, and the ceiling held, V100 sm70, 2026-09-03

> Status: **shipped; correct but not sufficient.** `DraftHead.forward` gained `last_only`, so
> the draft's readout is reduced to the one position per row that `_draft_step` actually reads.
> PyTorch's allocation at B=8 ctx=512 drops **27.39 → 25.86 GiB (−1.53 GiB)** and the OOM
> traceback no longer mentions it. But **ctx=512 still OOMs**, now in
> `paged_attention_split`'s partials at **1.50 GiB** — the same allocation recorded for the
> B=4 ceiling. At ctx=32 the tick goes **233.8 → 228.2 ms (1.8% faster)** with `tok/forward`
> **identical at 14.00**.

## Context

[B=8's ceiling was located](2026-09-03-b8-ceiling-is-the-draft-f32-readout.md) by letting the
run fail with its traceback: `_draft_step → DraftHead.forward → linear_fp4` allocating a
vocab-wide f32 `y2` at prefill width. `_draft_step` reads one row per request out of it —
**8 rows of 248320 = 7.6 MiB against 1444 MiB allocated, 191×**.

**The trunk had already solved this exact problem.** `model.py:371` carries the note "lm_head
over every prefill position is 4.7% of the FLOPs and a 508 MB output thrown away" and takes a
`last_only` argument, a per-row valid-length list, precisely because a device-side length
lookup would be a host sync illegal inside a graph capture. `engine.py:745` passes
`last_only=seq_q` on prefill ticks and `False` only for verify, where every chain position is
genuinely read. The draft's `forward` simply lacked the parameter — so the fix is the trunk's
own mechanism, not a new one.

## The change

`spec.py`: the same reduction, placed **after** `hidden_out.append(x)` so the chain's
full-width hidden indexing is untouched, and **before** the norm and lm_head so the readout
is never materialised wide.

`engine.py`: `last_only=sq` at the prefill/verify call site; the caller's read becomes
`logits[:, :1]`. The chain-loop call at `engine.py:975` is already T=1 (`reshape(-1, 1)`,
`seq_q_lens=ones`), so the `x.shape[1] > 1` guard skips it and it is left alone.

## Results

`bench_ctx_decode.py --depth 3 --batch 8 --tokens 32`, one process per context.

| | before | after |
|---|---:|---:|
| ctx=32 tok/s | 60.1 | **61.2** |
| ctx=32 ms/token | 16.7 | 16.3 |
| ctx=32 **tok/forward** | **14.00** | **14.00** |
| ctx=32 tick (ms/tok × tok/fwd) | 233.8 | **228.2 (0.976×)** |
| ctx=512 | OOM: **1.41 GiB** at `linear_fp4` `y2`, 12.38 MiB free | OOM: **1.50 GiB** at `paged_attention_split`, 396.38 MiB free |
| PyTorch allocated at failure | 27.39 GiB | **25.86 GiB (−1.53)** |

**`tok/forward` identical to the hundredth** is the parity evidence that matters: acceptance is
a function of which token the draft proposes, so selecting a different row would move it. 14.00
in both runs, over 8 requests × 32 tokens × 3 draft positions — roughly 768 draft decisions
with bit-identical acceptance counts, on the real 27B head rather than tiny weights.

## What the second OOM says

The readout is gone (−1.53 GiB allocated, absent from the traceback) and the ceiling did not
move. Two facts explain the gap between −1.53 GiB allocated and only +384 MiB free:

1. The two runs fail at **different points in the tick**, so "free at failure" is not a fixed
   baseline — the second got further and had allocated more by then.
2. **Reserved-but-unallocated went 1.67 → 2.82 GiB.** The allocator kept most of the freed
   space as cache rather than returning it, so a fresh 1.50 GiB request could not use it.

So at ctx≥512 the binding constraint is `paged_attention_split`'s partials, which scale with
`B × heads × splits × history` — and **1.50 GiB is the same figure already recorded for the
B=4 ceiling** in `bench_ctx_decode.py`'s own `--max-ctx` help text. B=8 has been pushed back
to exactly the B=4 wall, not past it.

Whether that wall is footprint or fragmentation is a separate, testable question: 2.82 GiB
sitting in reserve while a 1.50 GiB request fails is what `expandable_segments:True` exists
for, and torch names it in the error text. That test is running with its prediction committed;
this entry does not assume its outcome.

## Rule

**A negative control that passes is not a control.** My first check flipped the reduction to
row 0 and all four draft-parity tests still passed — instrumentation confirmed the branch
*executed* (T=6, 7, 8 with `last_only=True`), so the path was covered and the **input** was
not: that parity spy reads `out[i, -1]` *after* the reduction, comparing a wrong index against
whatever it chose. The replacement drives `DraftHead.forward` twice on identical input and
asserts the reduced readout equals the full-width one at the row it claims, with a prior
assertion that the positions differ at all — without which the test is unfalsifiable.
Negative control verified: row 0 fails it, the last row passes.

Second: **a committed prediction can be right about the number and wrong about the
mechanism.** I predicted ctx=32 unchanged "because a decode tick is T=1, so the reduction is a
no-op". It is T=1+depth=4, the reduction fires on verify ticks too, and the 1.8% is real. The
number landed close to the prediction for a reason I had wrong, which is not a confirmation.

Third: **stop inverting a byte count into a shape.** Three attempts, three failures: an
assumed vocab of 151936 (real: 248320), then a printed M=2491 narrated as 2611, then a prefill
model predicting 3880 MiB against a measured 1444. `max_num_batched_tokens=512` is a *whole
tick's* budget, not per request. The traceback's file, line, and call chain were sufficient
for both the diagnosis and the fix; the decomposition was never needed.

## Gate

191 tests pass, ruff clean. New test
`tests/test_e2e.py::test_the_draft_readout_reduction_picks_the_last_valid_row` with its
negative control verified. End-to-end acceptance identical (`tok/forward` 14.00). GPU verified
idle before each launch.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **draft readout, rows read vs allocated** | **8 of ~1524 — 7.6 of 1444 MiB, 191×** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **PyTorch allocated at B=8 ctx=512** | **27.39 → 25.86 GiB (−1.53)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | B=8 ctx=32 tick | 233.8 → **228.2 ms (1.8%)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | B=8 ctx=32 tok/forward | **14.00 → 14.00 — identical** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=512 after the fix** | **still OOM — 1.50 GiB, paged_attention_split** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | that 1.50 GiB vs the B=4 ceiling | **the same allocation — no progress past it** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | reserved-but-unallocated at failure | 1.67 → **2.82 GiB** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | first negative control (row 0) | **passed all 4 parity tests — void** |
