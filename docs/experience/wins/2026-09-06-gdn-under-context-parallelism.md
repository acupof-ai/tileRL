# GDN under context parallelism: zigzag chunks, a conv halo, and an affine prefix scan — cpu, 2026-09-06

## Context

Context parallelism splits a sequence across ranks. Full attention needs only the
KV of other ranks; the gated-delta layer is recurrent, so a chunk cannot start
until it knows the state the chunk before it produced — and that chunk lives on
another rank.

`init_tp`'s cp-only defect had to be fixed first
([entry](2026-09-06-a-cp-only-mesh-inherited-the-world-as-tp.md), #182); it
dominated every number here and is not repeated below.

## What zigzag costs the recurrence

Under zigzag a rank's tensor is **not one contiguous span**. Rank 0 of 2 holds
chunks 0 and 3, so it must be split and each half handled at its own sequence
position — two cross-rank dependencies, both keyed by **chunk** and never by rank:

- **The conv halo**, `kernel - 1` rows of the chunk before this one in sequence
  order. Dropping it corrupts the **whole** chunk, not just its first rows, because
  the bad k/v feed the recurrence. The gate's own `--no-halo` arm, run here: chunk 3
  reads **1.2e+00** on the first 3 rows and **5.3e-01** on everything after them,
  chunk 1 3.1e-01 / 1.5e-01, chunk 2 2.9e-01 / 7.2e-01. Chunk 0 stays at 0.0e+00,
  having no predecessor to take a halo from. **The "rest" figure exceeding the
  "first3" figure on chunk 2 is the point** — a spot check of the boundary rows can
  look like the smaller error.
- **The incoming state**, from an affine prefix scan over each chunk's `(A, B)`.

Both dependencies are resolved by shipping the chunk id alongside the payload and
sorting on it. `affine_prefix_scan` all-gathers `(A, B, id)` and sorts by the id
with the comment *"the gather returns rank order, and only the ids say what the
sequence order is"*; `cp_halo` builds an `owner` map from chunk id to
`(rank, index)` and takes each chunk's predecessor through it. Neither reads a
tensor at `parts[rank]`. Under a contiguous split rank order and sequence order
coincide, so the rank-keyed spelling would be correct there and produce
correctly-shaped, wrongly-paired tensors here — which is why the id travels with
the data rather than being recomputed from the rank.

## Gates, and what each one is for

| gate | asserts | worst measured |
|---|---|---|
| `gdn_cp_scan.py` | a chunk started from its composed prefix matches the sequential scan | out 4.38e-07, state 2.99e-07 |
| `gdn_world2.py` | cp=2 zigzag, every chunk from its scanned prefix | out 3.2e-07, state 1.4e-07 |
| `gdn_halo_world2.py` | every chunk with its left context matches the sequential run | first3 0.0e+00, rest 0.0e+00 (all 4 chunks) |
| `cp_model_world2.py` | a split forward matches the unsplit one, both layer kinds | logits rel 8.8e-07 (r0), 9.7e-07 (r1) |

The halo gate reading exactly 0.0e+00 on all four chunks is expected rather than
suspicious: with the correct left context the conv inputs are bit-identical to the
sequential run, so the only difference would be reduction order, and there is none
inside a depthwise conv row.

`cp_model_world2.py` is the only one of the four that reaches the residual add, and
it is what caught the `init_tp` defect the three op gates could not see.

**None of the three new gates was in CI, and the step's own floor could not see
that.** `ci.yml` listed six gate paths by hand and skipped any that was absent
(`[ -f "$gate" ] || continue`), with a floor of `ran > 0` — so the step passed at
6 of 9 exactly as it would at 9 of 9, and a "phase exit" would have landed on
gates nobody runs. Found in review of this PR, not by any gate. Now globbed
(`tests/*_world[0-9].py tests/gdn_cp_scan.py`) with the `-f` skip removed and the
floor raised to `>= 9`: a new gate is picked up without an edit, an absent one
fails loudly, and an unmatched glob leaves the literal pattern which then fails on
its own. Both arms run: 9 on this tree rc=0, a two-file fixture directory rc=1
saying `only 3 gates, expected >= 9`.

## The composed prefix is not rounded here

`_gdn_cp` computes `a_pre[i] @ state.float() + b_pre[i]` and passes it on **without**
`.to(state.dtype)`. The round was there first, and it diverged from the sequential
path: with it, `cp_model_world2.py` reads **rel 3.0e-05** against **8.8e-07**
without — 3x over the gate, on the CPU arm.

The reason it is not model.py's decision to make is that each backend arm marshals
its own operand. On the CPU arm `reference.gdn_chunk_core` takes `_f32(state)`; on
sm90 `_gdn_wy_core` passes `_bf16(state)`, with an existing comment at
`backend.py:1243` saying the scan's gemm operand is bf16 so the state rounds on
entry either way. A caller that rounds first is redundant on one arm and lossy on
the other.

## Controls

`__pycache__` cleared before each, and each file restored by copy rather than
`git checkout`.

| reverted | goes red |
|---|---|
| the f32 prefix (round back in) | `cp_model_world2.py` at rel 3.0e-05 |
| the `init_tp` fix (#182) | `cp_world2.py` assertion, and `cp_model_world2.py` at rel 1.2e+00 |
| the halo (`--no-halo`, the gate's own arm) | `gdn_halo_world2.py`, printing `correctly FAILED` |

## Not established

- **Every number here is the CPU arm.** On sm90 the WY scan rounds the composed
  prefix on entry, same as the sequential path at every `_WY_CHUNK` boundary; the
  gates' tolerances have not been run there.
- **cp=2 only.** Nothing above says the zigzag indexing is right at cp=4 — the
  chunk-id math generalises by construction and was not measured.
- **Forward and backward through the gates, no training run.** No loss curve, no
  step-time number, and no throughput claim for CP at all.
- **CP is training-only and refuses serving** (`_refuse_cp_serving`); the paged
  pool holds whole sequences. Untested against a serving engine because the
  refusal is what is tested.

## Rule

An op gate cannot see a defect in the layer that calls it. Three gates here held
the recurrence to the sequential scan at 1e-7 while the model's residual was wrong
by its own magnitude, because none of them reaches `_add_via`. When a gate suite is
green and the end-to-end number is not, the missing gate is at the layer the suite
does not enter — not a tighter tolerance on the ones that pass.
