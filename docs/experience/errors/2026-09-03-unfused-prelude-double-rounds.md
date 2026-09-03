---
question: Why does `fuse_projections=True` change 53 of 1000 MMLU answers, and which arm is right?
source: H20 sm90 card 6, tilelang 0.1.13, torch 2.11.0+cu129, 27B NVFP4, main ecc68e0, scripts/probe_fusion_weights.py + probe_fusion_kernels.py + probe_attn_prep.py + probe_layer_amplification.py
---

# Fusion is not the defect: the unfused prelude double-rounds, and it is `cli.py`'s default

**The 53 flips are the UNFUSED arm being wrong.** `attn_prep` is closer to a
dense f32 norm+rope on **580 of 580** differing K-plane elements, by a factor of
**2.0007** — exactly one extra bf16 rounding step. The fused path keeps norm+rope
in f32 registers and casts once at the store; the discrete path rounds at
`rmsnorm`, again at `rope`, and again at `write_tokens`.

So the 746/1000 measured with fusion on is the more accurate score and the
742/1000 from `cli.py` is the degraded one. Every `uv run tilerl` eval has used
the degraded arm, because `cli.py` defaults `fuse_projections=False` and
`scripts/mmlu.py` passes True.

## The chain, seed to logit

| step | measured | probe |
|---|---|---|
| `_fuse_projections` concat | bit-for-bit exact, 5/5 groups, 0 declined | `probe_fusion_weights.py` |
| fused vs unfused GEMM | **bitwise identical**, M=1/8/64 | `probe_fusion_kernels.py` |
| prelude divergence, K plane | 1.562e-02 max, 1.44e-04 rel, 580/262144 elements | `probe_attn_prep.py` |
| which arm is closer to exact | **attn_prep, 580/580, ratio 2.0007** | `probe_attn_prep.py` |
| first divergent layer | **3** — the first full-attn layer; 0-2 bitwise identical | `probe_layer_amplification.py` |
| amplification to the logits | 7.738e-02 → 4.340e+01, **x561**, floor exactly 0.000 | `probe_layer_amplification.py` |

`fuse_projections` is not a fusion toggle. `model.py:202`'s
`if self._has(qkv_key)` is true only when fused, and that branch calls
`backend.attn_prep` — norm + rope + KV-write in one launch — then returns before
the discrete epilogue exists. The qkv concat is only its trigger. `attn_prep` has
no other caller.

## The fix: decided, not landed

**Cast once in the discrete path. Do not change which branch runs.** Recorded here
so the choice is not re-derived:

- Flipping `cli.py`'s default routes every eval through `attn_prep`, which is
  sm90-only with no CPU twin — the CPU cell would keep double-rounding and the two
  targets would disagree by construction. Fixing the arm makes both targets right;
  switching branches makes one right and hides the other.
- Smaller blast radius. `attn_prep` returns early past `paged_attention`, so
  flipping the default moves the kv-write path, the gate slice and the attention
  entry at once. Casting once changes rounding in one place.
- **The discrete path is what training differentiates.** This is not only an
  eval-accuracy question — it is the arithmetic the tape runs backward through.

Unlanded deliberately: serving-path arithmetic wants its own tranche with a parity
gate, and it should land beside a CPU twin for `attn_prep` rather than alone.

## Why the parity gate could not catch it — two independent holes

**Hole 1: the branch has no CPU twin.** `src/tilerl/testing.py:53`:

    def attn_prep(self, *args, **kwargs):
        return None

`RefBackend.attn_prep` is a stub returning `None` unconditionally, so
`model.py:212`'s `if qn is not None` is always false on the CPU target and the CPU
cell always takes the **discrete** path. Both arms of every CPU parity test run
the same prelude, so the gate compares the wrong arm against the reference and
passes. `attn_prep` is reachable only on sm90, only through the fused branch.

**Hole 2: the extra rounding is not in the CPU cell at all.** The rounding is
`kernels.py:110`'s `Y = T.empty((M, N), "bfloat16")` in `make_rmsnorm_fused_bf16`,
and `registry.py` registers the bf16 norm outputs **only in `_SM90_KERNELS`** —
`_CPU_KERNELS` maps `rmsnorm_apply` to `make_rmsnorm_apply`, which emits f32
(`kernels.py:60`), and has no `rmsnorm_fused` at all. The CPU cell is f32
throughout and does not double-round.

So writing the CPU twin makes the *branch* comparable on CPU and still leaves the
*rounding* invisible there. **A CPU parity gate cannot catch this defect even after
the twin lands**; the gate has to run on sm90. Two holes in the same defect, and
closing the one that is easy to see does not close the other.

The chain, counted: discrete is f32 → **bf16** (`rmsnorm_fused`) → f32
(`Backend.rope` re-widens an already-rounded value, recovering nothing) → f32
(rope kernel emits f32, `kernels.py:425`) → **bf16** (`write_tokens`).
`attn_prep` is f32 → f32 → **bf16**. One extra rounding, which is the 2.0007x.

This is not a missing test. Writing a test would not have helped without the CPU
twin, and the twin alone would not have helped either. Same class as
[a stub that replaces the component under test makes the gate blind to it](2026-09-03-an-oracle-stub-blinds-the-gate-it-sits-in.md),
one level worse: there the stub replaced the component in one gate, here it is the
only CPU implementation there has ever been, so every gate inherits it.

## Where the extra rounding is observable

`backend.rmsnorm` has six call sites in `model.py`, and the bf16 intermediate only
matters where the norm's output survives to a stored value rather than being
requantized immediately:

| site | consumer | bf16 intermediate matters? |
|---|---|---|
| `input_norm` (199, 270) | fp4/fp8 GEMM | no — fp8 quantization rounds harder |
| `post_attn_norm` (337) | fp4/fp8 GEMM | no — same |
| `q_norm`/`k_norm` (239, 240) | rope → bf16 KV pool | **yes — this is the measured 2.0007x** |
| `final_norm` (390) | lm_head `_linear` | **unmeasured, and it is the logit path** |

`final_norm` is the open one. The argument that a bf16 intermediate is harmless
where the result is requantized does not obviously transfer to the site whose
output *is* the logits, and the flips being chased are logit-margin flips. It may
be dominated by the head's own quantization; nobody has measured it. An f32-output
variant covering q/k norm only would leave that site as it is.

## Four hypotheses that were wrong, and how each was killed

**`max|d|` as the ranking statistic.** It printed `1.559e-02` vs `1.559e-02` — a
tie to four significant figures — and the "neither arm is wrong, two valid
roundings" conclusion was already written. Both arms round to the same bf16 grid,
so the largest error is the quantum at the largest-magnitude element and it is the
same element in both arms. Max measured the grid, not the arithmetic. Mean over
the differing elements plus a per-element win count separates them 580-0. This one
nearly produced the exact opposite conclusion.

**N-dependence in plan selection.** Fusing changes N — 12288+1024+1024 served as
one GEMM instead of three — so the arms land on different tilings. Stated with a
correct code citation (`Backend.linear_fp4` calls `self._plan("linear_fp4", M, N,
K)` and pads N to the plan's `Np`) and no measurement, and **it produces no
numerical difference**: 14336 vs the three separate Ns match bit-for-bit. The
citation was right and the inference from it was wrong. "Changes N" needed "and the
tiling changes the result".

**Partial-rotary pairing.** Qwen3.8 is head_dim 256, rotary 64, so the two arms
looked like they must pair different dims. Refuted by arithmetic before spending a
run: `Backend.rope` slices `x[..., :64]` and its kernel derives `half = D//2`
**from the slice** (32), and `attn_prep`'s `RD2` is `len(InvFreq)` (32). Same
pairs, same untouched tail.

**The hidden-state capture, twice.** First version kept the tensors `hidden_out`
returns and copied after the forward — they are views into buffers later kernels
reuse, so all 64 entries held the last write. Floor came out **2.767e+02**
comparing an arm against *itself*. Fixed with copy-on-append; the floor got
*worse*, **1.173e+03**, because the prompt is 76 tokens and the prefill tick runs
**T=128**, so rows 76..127 are never written and hold allocator garbage.

## The controls that did the work

- **Negative control on the GEMM comparison**: perturbing one scale element by
  1.5x moves the output 6.007e-02. Without it, "bitwise identical" is
  indistinguishable from a comparison that cannot see anything.
- **Noise floor on the amplification trace**: the same arm run twice, asserted
  `== 0.0`. Both capture defects were caught by it, the second one entirely by the
  assert rather than by reading. It started as a printed warning; a warning is
  what lets a broken capture emit a plausible table.
- **Positive control, unplanned**: gate and V plane come out bitwise identical
  because both are pure copies through the two arms. That retires the
  "do the arms disagree about which half of the interleaved `[query; gate]` block
  is which" question without a separate run.

## Correction to the 09-03 concurrency entry

[`errors/2026-09-03-mmlu-score-depends-on-concurrency.md`](2026-09-03-mmlu-score-depends-on-concurrency.md)
and its CHANGELOG line state the mechanism as: concurrency sets B, B sets
`M = B*W`, `M` picks the fp4 arm, so two concurrencies run two reduction orders.
**Measured false.** Concurrency 8 and 32 return 742/1000 each with **0 of 1000
answers differing**. `M` on a prefill tick comes from `_PREFILL_BUCKET` (64) and
the prompt's own padded length, not from the batch — concurrency batches
independent rows and each keeps its own width. `M = B*W` is a decode-tick
mechanism and MMLU at `max_new_tokens=1` runs no decode ticks. The variable that
actually differed between the two callers was `fuse_projections`, which is this
entry.

The fix recorded there — `mmlu_accuracy` returning the concurrency it used — still
stands. Recording the parameter was right; the reason given for why it mattered
was not.

## Rule

A summary statistic can measure the harness instead of the system, and it looks
like a result either way. Three of the four falsified hypotheses above died that
way: comparing products where the question was about weights (measured cuBLAS),
comparing padded rows (measured the allocator), ranking by max where both arms
share a rounding grid (measured the grid). In each case the number was plausible
and the conclusion was wrong.

The tell is a value that is too clean: `1.431e-06` for a "concat bug",
`1.559e-02` twice for two different arms, a floor that matches the signal while
the logits are bit-identical. Before reading a number, ask what it would read if
the thing under test were absent — that is the negative control, and it is cheaper
than the wrong conclusion.

A filter that silently drops the cases the hypothesis is about looks exactly like
a run that checked everything. The first weight check gated on
`all(f"{k}.wq" in params)` and skipped every fp8 group — `qkv` and `qkvz`, the two
largest and the two the hypothesis named — printing three rows and no mention of
the two it never reached. Print the skips as a third outcome.
