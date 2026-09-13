# sm90 fused attn_prep corrupts sparse K/V when >1 ragged row shares a packed prefill tick

> Status: **open.** #563 lands the guard (sparse forces the unfused writer);
> the fused twin is not fixed. Under the guard sparse k=128 matches dense on a
> matched question set: spec-off B=8 MMLU, `--n 400 --first-n 400 --seed 0`,
> sparse 0.865 (389 tok/q, 32.6 tok/s) vs dense-fused 0.858 (397 tok/q,
> 187.0 tok/s), `/work/65-guard563-s400.json` vs `/work/cc-dense400.json`,
> card 0. The earlier "0.7986 vs 0.915" reading compared two different question
> sets (`--n 400` vs `--n 2000 --first-n 400`; the harness samples per n). What
> the guard costs is speed, 94 → 33 tok/s, so sparse stays opt-in
> (DEFAULT_SPARSE_K=0) until the fused twin is fixed. The fused twin PR removes
> this status and the OPEN.md row.

## Context

65's paired MMLU (spec OFF, sparse k=128, H20 sm90, B=8, ~400-token prompts) scored
~0.45 vs dense 0.915, with sparse streams degenerating into repetition loops.
Sparse k=128 + the 8-page own window covers these whole prompts, so the attention
SET is dense-exact; a divergence there is a structural defect, not a
k-too-small fidelity trade. Hotfix #558 made sparse opt-in; this entry names the
cell and the guard that makes opt-in correct.

## Root cause

The sm90 fused `attn_prep` (`kernels_mma.make_attn_prep` / `make_attn_prep_fp8`)
does q/k norm + RoPE + the paged K/V pool write in one launch off the fused-qkv
GEMV. With more than one ragged sparse row sharing a packed-prefill tick it
corrupts K/V. The unfused path (`rmsnorm_f32`/`rope` + `write_tokens`) is exact.

## Measured (card 5, H20 sm90, ab59505e, 2026-09-12)

Same five MMLU prompts (lens 88/102/115/84/70, 459 tokens total, all admitted
to one B=8 prefill batch), harness `run_arm` verbatim, one arm per process,
first-generated-token logits saved keyed by prompt tokens so dense/sparse pair
on identical inputs:

- fuse_projections=1 (serving default): dense-vs-sparse g0 max_abs
  7.98 / 9.73 / 11.30 / 9.69 / 8.31, mean ~1.1; two argmax flips; even
  non-flip rows have their top-2 margin perturbed 2-7x.
- fuse_projections=0 (unfused): all five max_abs 0.0, mean 0, argmax equal.

Controls (all bit-exact dense-vs-sparse, max_abs 0): each prompt alone at B=1
native and padded B=8; forced-think on/off; full-prefix second submit. On sm70
65 fed the identical 115-id sequence B=1 and got max_abs 0. So the defect needs
>1 different rows in one fused sparse packed prefill tick and is sm90-only.

The g0 argmax alone is not a reliable signal: the top-2 margin on the 115-token
question is ~0.14, a near-tie, and dense's own argmax moves across padding and
arch. The matched full-logit max_abs (8-11) is the proof; the split that names
the cell is fuse on vs off.

## Fix (guard)

`build_engine` sets `backend.no_fused_attn_prep = True` whenever `sparse_k>0`;
`Backend.attn_prep` then returns None on EVERY tick (prefill and decode), so
`model.Model` slices the fused qkv and takes the exact unfused `write_tokens`
path for both. Dense keeps the fused prep. The attention kernel is otherwise
untouched. The fused twin is fixed properly in a follow-up with a served-dims
sm90 pending-remote parity gate (B=2 ragged sparse, fuse=1, dense-vs-sparse
max_abs 0).

## Rule

A fused prep kernel that fuses a per-row paged K/V write must be gated under
sparse packing until it has a B>1 ragged-row parity gate; the unfused writer is
the safe fallback. A near-tied argmax flip is not evidence — compare matched
full logits.

The guard trades speed, not accuracy: on the same 400 questions sparse k=128
spec-off reads 0.865 vs dense-fused 0.858 (card 0, `--n 400 --first-n 400
--seed 0`), at 32.6 vs 187.0 tok/s. K/V pages written by the fused and unfused
twins are byte-identical against an f64 reference (kvtwin/kvpre probes, decode
and prefill ticks); only Q differs by one bf16 ulp from f32 reduction order.
Two MMLU jsons are comparable only if their `gold`/`subjects` lists agree:
`--n` changes the sample, not just `--first-n`.