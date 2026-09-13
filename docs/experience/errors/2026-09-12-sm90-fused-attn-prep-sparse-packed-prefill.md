# sm90 fused attn_prep sparse path attended over the own-window table, not the packed sparse table

> Status: **fixed.** #563 guarded the symptom (sparse forced the unfused writer);
> this fix corrects the cell — the fused branch's attention read — and removes
> the guard, so sparse serves the fused path again. The K/V-writing kernel was
> never wrong.

## Context

65's paired MMLU (spec OFF, sparse k=128, H20 sm90, B=8, ~400-token prompts) scored
~0.45 vs dense 0.915, with sparse streams degenerating into repetition loops.
#558 made sparse opt-in; #563 measured the divergence onto "fuse on vs off" and
forced every sparse tick through the unfused `write_tokens`, on the hypothesis
that the fused K/V writer corrupted pages for >1 ragged packed prefill rows.

Two later measurements reopened that diagnosis:

- a symmetric decode cut (writer vs geometry) showed sparse packed geometry is
  bit-exact on sm90 at B=1 and B=8 — sparse-unfused vs dense-unfused max_abs
  0.0 across g0..g3 — while dense-fused vs dense-unfused alone moved 0.3-5;
- a served-dims kernel probe called `make_attn_prep` directly (B=2
  ragged/equal/padded/page-cross, nonzero `page_base`/absolute `seq_len`
  sparse geometry) and the written K/V matched the f64 reference and the unfused
  writer: V 0.0, K ≤ 0.002 (bf16-boundary reduction noise), no B>1 dependency.

## Root cause

`Model._full_attn` has two attention-read paths. The unfused path builds
`SparseForward.attention_args` — the packed `[selected earlier ; own]` block
table with `seq_len = n_sel*16 + own_len` — and hands it to `paged_attention`.
The fused early-return (taken when `backend.attn_prep` returns a non-None `qn`,
i.e. sm90 fuse=1) called `paged_attention` with the raw `kv.block_table` /
`kv.seq_len`. On a sparse `BatchKv` that table is the own-window-only table, so
the fused path attended over just the trailing window and never ran selection:
a routing defect in `model.py`, not a defect in the `make_attn_prep` kernel.

The K/V pages both paths wrote were the same; the gap was entirely in which
pages attention read. It was also not B>1-gated: sparse-fused vs
sparse-unfused at B=1 g0 read max_abs 7-17.

## Fix

The fused branch now builds the same sparse packed table the unfused branch
does:

```python
sf = getattr(kv, "sparse", None)
if sf is not None:
    block_table, seq_len = sf.attention_args(
        kv.kv_pool.plane_of(layer_idx), qn, h_idx)
```

`#563`'s `backend.no_fused_attn_prep` guard is removed (engine set + backend
attr + check); dense and sparse both use the fused prelude.

## Measured (card 5, H20 sm90, fix applied to 0f8b8ebf, 2026-09-13)

Same five MMLU prompts, one arm per process, matched full logits g0..g3, sparse
fuse=1 vs the unfused sparse baseline:

- B=1: all 20 row/step argmax equal; max_abs identical to the dense
  fused-vs-unfused cut (2.78/1.00/1.77/1.51 … 5.0) — the remaining delta is the
  two norm reductions' f32 accumulation order (fused single-order serial sum vs
  the unfused 256-thread tree), erased at the bf16 K/V store; pre-fix 7-29 with
  argmax flips.
- B=8 mixed: all 20 argmax equal, max_abs 0.38-4.96; pre-fix 2-27 with two
  flips.

The residual fused-vs-unfused logits difference is the documented Q/K
reduction-order precision difference, not a bug.

## Rule

A fused early-return is a second attention-read path: if the unfused path
derives its block table from a per-tick descriptor (sparse packed table), the
fused path must derive the same descriptor rather than reading the static
`BatchKv` table. A kernel that writes the pool correctly can still be blamed
for a divergence whose actual cell is the attention read — localize with a
direct kernel-level reference before naming the writer.
