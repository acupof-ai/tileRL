# Chunked attention removed: correct code, wrong term — H20 sm90, 2026-09-06

> Status: Reverted (#196 shipped and came out the same night)

## Context

A GRPO step OOMs at gen 4096, group 8, on one 95.22 GiB card. The diagnosis on
record was the training-path T² score matrix: `1.69 GiB × 16 full-attn layers =
27.1 GiB`, later corrected here to **10.16 GiB** (the tape saves q/k/v, never
`att`, so layer depth does not accumulate — peak live is 3x one tensor in
forward, 6x in backward). #196 removed that tensor: chunked, flash-style forward
and backward, 138x/111x drop in peak attention intermediates on the CPU target,
parity to 6.2e-07, 422 tests green.

**Neither figure was the binding term.** The premise is refuted, not refined.

## What happened

**gen 4096 still OOMs with chunked attention, and the traceback never enters
attention:**

```
prof_grpo_step.py:139 → train.py:312 rl_step → train.py:207 _step → train.py:155 run
  → model.py:481 forward → model.py:434 _mlp → autograd.py:45 checkpoint
  → model.py:447 _mlp_body → model.py:235 _linear → model.py:247 _base_linear
  → autograd.py:78 master_linear → backend.py:804 linear_fp4 → backend.py:461 _epilogue
torch.OutOfMemoryError: Tried to allocate 290.00 MiB. 95.22 GiB capacity,
121.56 MiB free, 95.10 GiB in use (91.71 allocated, 2.60 reserved-unallocated)
```

The failed allocation identifies itself. With `intermediate_size=17408`, one MLP
intermediate `[4352, 17408]` f32 is **289.0 MiB** against the **290.00 MiB**
requested. Chunked attention freed up to 10.16 GiB and the step still needed
95.10 of 95.22.

### And it costs 1.37x–1.45x on backward

gen 1024, group 8, LoRA-16, micro 1, card 6, same probe (`prof_grpo_step.py`,
sha256 `f1c4b6d6dd86` on every tree measured):

| tree | attention | backward s | step s | note |
|---|---|---:|---:|---|
| a16ff9c | dense | 71.53 | 131.60 | #192's warm mean |
| 36bfe6f | dense | 67.77 | 130.95 | the control, warm mean of 2 |
| 2cafc87 | chunked | **97.94** | 161.11 | warm mean of 3, spread 0.9% |

Against the control: backward **1.445x** (+30.2 s), step **1.230x** (+30.2 s).
Against #192: **1.369x**.

**The honest figure is a range, not a ratio.** The two dense runs differ by 3.76 s
(5.3%) on backward with identical code — though only 0.65 s (0.5%) on the step —
so chunked attention costs **1.37x–1.45x** on backward depending on which dense
run is the reference, and this entry gives both rather than the flattering one.
The direction is not in doubt: +30.2 s against a ±3.8 s spread.

### The control also exonerates #190

#190 (`_const_f32` refilling a cached buffer instead of rebinding) sat between
the two trees, so the first comparison had two changes in it. The backward window
contains 64 recomputed MLP forwards, so the cast path does run inside backward
time — a real reason to measure rather than dismiss. Measured: **36bfe6f vs
a16ff9c is 0.947x on backward and 0.995x on the step**, i.e. #190 alone moved backward by nothing outside noise.
`backend.py:1424`'s early return catches the 63 repeat calls; only the first
after a version bump copies.

## Controls

| control | reading |
|---|---|
| dense on the pre-#196 parent (36bfe6f), same probe sha | backward 67.77 s warm mean (68.26 / 67.77 per step) |
| chunked, three warm steps | 98.35 / 98.39 / 97.49, spread 0.9% |
| pod tree identity, read in the same call as the numbers | `stamp=36bfe6f chunked=0 dense=2` and `stamp=2cafc87 chunked=2` |
| revert completeness | all four files byte-identical to `766666b^`; zero references to the chunked symbols remain |

## What comes next, and what does not

- **The 4096 cap decision is unchanged**: unreachable at group 8 on one card,
  same as #192 concluded, for a different reason than #192 states.
- **The open measurement**: what holds 91.7 GiB during a gen-4096 *forward*. The
  OOM frame is inside a checkpointed MLP body, so per-layer intermediates should
  die on exit and something else accumulates. `checkpoint` records
  `args = (layer_idx, x, kv, backend)`, so each layer's **input** is retained by
  construction — `[4352, 5120]` f32 = 85 MiB per layer. Whether 64 of those are
  simultaneously live is a measurement nobody has taken, and a size × layer count
  is exactly the mistake this entry is about.
- **No sm90 TileLang cell.** It would speed up a path that buys nothing at the
  shapes we run.

## Rule

**Read the OOM traceback before optimizing for the OOM.** An OOM names its
allocation: a frame list and a byte count identify the tensor, and 290.00 MiB
matched one MLP intermediate to within rounding. Arithmetic about a different
tensor — however carefully measured — is not evidence that tensor is binding. My
own refcount-death correction of 27.1 GiB down to 10.16 was right and irrelevant;
refining a number inside a wrong premise makes the premise look better supported.

Corollary for this code specifically: **it is shelved, not wrong.** The
implementation is correct, gated and parity-checked, and PR #196's diff is where
to find it if a shape ever makes T² bind. Do not rebuild it from scratch.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-06 | 2cafc87 | H20 card 6 | cuda sm90 | 27B, chunked | n/a | n/a | backward 97.94 s/step |
| 2026-09-06 | 36bfe6f | H20 card 6 | cuda sm90 | 27B, dense | n/a | n/a | backward 67.77 s/step |

Raw artifacts: `/work/grpo1024.log`, `/work/grpo4096.log`, `/work/ctrl190.log` on
the pod.
