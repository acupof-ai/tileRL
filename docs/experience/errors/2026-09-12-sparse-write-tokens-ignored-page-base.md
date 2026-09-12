# Sparse prefill wrote own K/V to absolute page columns, ignoring page_base

Date: 2026-09-12. Target: sm70 V100 + sm90 H20 sparse prefill.

## Context

Production-path fidelity (the dense-vs-sparse `build_engine` harness, both
arms eager) on the 27B NVFP4 showed sparse prefill
diverging from dense even at k=ALL pages: 8k k=128/256/512 all KL ≈ 0.676 /
top1 ≈ 0.65, flat in k; 32k k=128 KL 1.050. A recall/mass effect must improve
as k grows and vanish at full coverage, so this was structural, not selection.

## Root cause

Sparse prefill's per-row block table is **page_base-relative**: it holds only
the chunk's own pages, and column 0 is logical page `page_base`. The torch
fallback `PagedKvPool.write_tokens` honors that — `block_table[pos//16 - base]`
(kv_cache.py) — but the tilelang writer kernels indexed the table with the
absolute logical column `BlockTable[b, pos//16]` and took no page_base:

- `write_tokens` (bf16, sm90), `write_tokens_f32` (sm70),
  `write_tokens_fp8`
- `attn_prep` (bf16, sm90 fused sparse-prefill prep), `attn_prep_fp8`

From the second prefill chunk on, `pos//16` is past the end of the own table
and reads zero padding, so the chunk's own K/V scattered into block 0 / wrong
frames while the real own frames (columns 0..own_pages−1, the blocks attention
reads) stayed empty. The first chunk (page_base=0) is correct, so divergence
started exactly at chunk 1.

The attention kernel itself was sound: vs
`RefBackend.paged_attention` at H=16 Hkv=4 D=256 64 pages, S=1/32/512 matched
to ≤1.5e-4. Cold precision (f16 vs f32), packed-table completeness/order, and
GDN recurrent state were all ruled out by direct probes.

Card evidence (8k k=512 f32 cold, plane 3, chunk 1): promoted-back selected
pages 0..31 byte-exact vs dense; freshly-written own pages 32..63 K max 8.42 /
V max 6.71. The bug was the own-span write, not history fetch.

## Fix

Thread a `PageBase[B]` int32 tensor through all five writer kernels and index
`BlockTable[b, pos//16 - PageBase[b]]`. The Python caller passes `page_base`
when the BatchKv carries one, else zeros, so the dense path is unchanged.

Red parity: `test_write_tokens_owns_page_base_relative_table` at served dims
(Hkv=4, D=256, chunk 512, page_base=32) asserts each own token lands in the
column-relative frame and that nothing leaks into padding columns. Green on the
CPU fallback cell; on sm70 the unfixed kernel failed it (worst 4.05) and the
fixed kernel passes (0.0).

## Rule

A kernel that reads a page table must be told whether column 0 is logical
page 0 or a sparse own span base; an absolute index silently reads padding.
Full-coverage fidelity (k = all pages, KL ≈ 0) is the gate that catches a
write-path bug that k-vs-k recall sweeps cannot — the KL was flat in k because
the own pages were wrong at every k.
