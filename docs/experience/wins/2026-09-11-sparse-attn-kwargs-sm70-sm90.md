# Sparse paged_attention kwargs port to sm70/sm90 — V100 + H20, 2026-09-11

> Status: pending-remote (GPU parity green; stacks on Unit F's RefBackend twin + engine)

## Context

Unit F adds sparse-KV (Quest) attention: each full-attn row attends to a set of
*selected earlier pages* (unconditionally visible) followed by its *own span*
(the chunk/window it just wrote, causal-masked by global position). It extends
`paged_attention` with five kwargs — `page_sel/n_sel/own_lens/own_offsets/q_start`
— and ships a CPU oracle in `RefBackend`. The tilelang sm70 (V100 f32 split-KV)
and sm90 (H20 bf16 MMA) cells raised `NotImplementedError` until this port.

## What Worked

**No new attention kernel.** All three tilelang cells — the C cell, sm70
`paged_attention_split`, sm90 `paged_attention_mma`/`decode` — mask purely by
*packed key-slot position* against `seq_lens - seq_q_lens (+t)`; none read an
absolute key position. So sparse is one host-side table remap in
`Backend.paged_attention`: per row build a concat block table
`[page_sel selected pages ; own span pages]`, packing the own span right after
that row's `n_sel` (rows are ragged), and set effective
`seq_lens = n_sel*block + own_lens`. The existing dense kernel then computes
exactly the twin's one-softmax `[selected unmasked ; own causal]`:

- selected slots occupy `[0, n_sel*block)` and precede every query → all-ones mask;
- own slot `j` has global position `own_offsets+j` and is visible to query `t`
  iff `own_offsets+j <= q_start+t`. This folds to the slot test
  `p <= hist+t` iff `own_lens - seq_q == q_start - own_offsets`, which the
  prefill geometry (`own_lens = q_start - own_offsets + seq_q`) and decode
  (`seq_q=1`) both satisfy. A ValueError guards the identity rather than silently
  mis-attending.

`own_offsets`/`q_start` never reach the kernel; only `n_sel`/`own_lens` drive the
remap. fp8 scale indexing follows the *physical* block id in the gather, and both
table halves carry physical ids, so the fp8 path needs no change either (gated off
in Unit F's first cut, which raises on an fp8 sparse pool).

**Right-pad rule.** Physical block id 0 is a genuinely selectable frame, so the
concat columns are never masked by id equality — `n_sel` (and the seq_lens bound)
is the sole validity marker. The own span is packed per-row after `n_sel`, not at
a fixed `Pmax` column, so a short row never reads a padded selected slot as real
unmasked data.

**Two twin bugs found while porting** (reported to the F author): the
`RefBackend` sparse `gather()` (1) reshaped `[nblk,Hkv,16,D]` without the
`permute(1,0,2,3)` the dense arm uses, scrambling block/head/token axes, and
(2) `unsqueeze(1).expand(-1,rep,-1)` on the resulting 4-D tensor crashed. Both
fixed to mirror the dense arm.

## Rule

A slot-causal paged-attention kernel serves sparse prefix+own attention with zero
kernel changes: remap the block table to `[selected ; own]` and set seq_lens to
the packed length. The free ride is exact only while `own_lens-seq_q ==
q_start-own_offsets`; guard that identity.

## Results

| date | machine | target | case | max rel vs RefBackend |
|---|---|---|---|---:|
| 2026-09-11 | Mac CPU | tilelang C | prefill overlap + decode, ragged, phys-id-0 selected | 3.1e-7 / 2.8e-7 |
| 2026-09-11 | V100 | sm70 f32 split-KV | same probe | 3.2e-4 / 1.9e-4 |
| 2026-09-11 | H20 card 0 | sm90 bf16 MMA | same probe | pending |

Dense (None sparse kwargs) path unchanged: `tests/test_ops_parity.py -k attn`
3 passed. Gate: `scripts/probe_sparse_attn_kwargs.py`.
