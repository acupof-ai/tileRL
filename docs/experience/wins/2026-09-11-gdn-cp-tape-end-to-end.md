# The GDN context-parallel tape runs end to end (conv kernel 1) — CPU, 2026-09-11

> Status: Shipped for the affine state transfer; cross-rank conv-halo adjoint pending (next PR).

## Context

Context-parallel training splits one GDN layer's sequence into zigzag chunks, each
started from an incoming state `s_in = a_pre @ state + b_pre` supplied by an affine
prefix scan over every chunk's `(A, B)` span map. The forward existed. The tape did
not: `gdn_span_ab_raw` had no reverse, `cp_prefix_scan` was absent from `autograd._BWD`,
and the `a_pre @ state + b_pre` start was a bare matmul+add the recorder never saw, so
under CP the state gradient was silently absent — no crash, a plausible loss.

## What worked

The span, the scan and the per-chunk forwards are one backend op (`gdn_cp`), so the
whole affine transfer records as a single tape entry with one matching reverse
(`gdn_cp_bwd`), instead of teaching the id-addressed tape about tuple outputs and raw
matmul/stack glue. The reverse sums two paths on the shared raw inputs: the ordinary
per-chunk `gdn_backward` (token output + its `dS_in`), and the state path
`dS_in -> (dA,dB) -> affine_prefix_scan_bwd -> gdn_span_ab_raw_bwd`. The prep forward
and its adjoint were extracted once (`_gdn_prep_save` / `_gdn_prep_backward`) and are
shared by `gdn_backward` and the span reverse.

The span reverse's one new primitive is the cotangent of a chunk's span operator
`a_i = exp(glast) I - R^T W` at a zero start state; `B_i = s_next` there, so the
existing `_gdn_chunk_bwd` carries it with three added terms.

Gradchecks (central differences, floats over a spawn Queue — tensors EOF-error on
Linux):

| gate | what it pins | green | red control |
|---|---|---:|---|
| `test_gdn_span_ab_gradcheck.py` | span `(A,B)` -> prepped inputs | 1.6e-4 | decay-scalar `a_i` 0.57 |
| `gdn_cp_tape_world2.py` | world2, real Tape+RecordingBackend, all 11 leaves | 4.3e-4 | decay-a 1.3e-2; no-scan 0.28 |

The world2 oracle is a single-process virtual CP that runs the four chunks from their
sequential exclusive prefixes and differentiates the GLOBAL loss summed over ranks;
replicated leaves (state, params) sum both ranks' tape grads.

**Not yet covered:** the tiny fixture is conv kernel 1, so every chunk starts from a
zero halo and only the affine transfer is cross-rank. The 27B runs kernel 4; under it a
chunk's prep mixes the prior chunk's last 3 raw qkv rows exchanged by `cp_halo`.
`gdn_cp_bwd` raises on a nonzero window rather than guess, so q/k/v under kernel 4 are
NOT gradchecked until the follow-up: a window-aware prep reverse plus a `cp_halo`
reduce-scatter that adds the window grad onto the other rank's q/k/v.

## Rule

A forward built from raw torch glue around an unregistered collective has no gradient
and no error — the gate is a world2 numerical gradcheck through the real tape against a
single-process sequential oracle, with a wrong-reverse control that goes red. A relative
metric on a near-zero grad component is noise; compare by vector RMS-rel.
