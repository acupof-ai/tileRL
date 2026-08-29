# Chunked gated-delta backward: 491K kernels a step -> 191K — 2026-08-29

> Status: Shipped. `_GDN_CHUNK = 16`.

## Context

The 27B train step ran 41 tok/s against a 2218 tok/s forward on the same model.
`reference.gdn_backward` looped over the time dimension twice in Python — a
forward re-scan and a reverse scan, ~28 launches per step per GDN layer — so
one step was **491,113 kernels, 62% of them micro-ops**, against 354 ms of
actual fp4 backward GEMM
([errors/…-train-step-is-the-gdn-per-step-loop.md](../errors/2026-08-29-train-step-is-the-gdn-per-step-loop.md)).

## What Worked

The adjoint of the chunkwise-WY form, hand-derived and landed as
`_gdn_chunk_fwd` / `_gdn_chunk_bwd`. The sequential dimension becomes
`t/CHUNK`; only chunk-START states are stored and each chunk's interior is
recomputed from one in the reverse pass, so `states`, `ps` and `deltas` — three
per-step tensors — are gone.

Same script, 64 layers, 1x256, H20 GPU 7:

| | before | after |
|---|---:|---:|
| GPU-busy | 2296.2 ms | **1491.3 ms** (1.54x) |
| kernels | 491,113 | **191,449** (2.57x fewer) |
| `linear_fp4_bwd` share | 15.4% | **23.7%** |

The real work is now the largest line in the profile, which it was not before.

End to end, `bench_harness --suite train` on the 27B (LoRA rank 16), every row
auto-raised. The win grows with T, which is the shape chunking predicts — the
launch count is the term that scaled with the sequence:

| B x T | before | after | | peak |
|---|---:|---:|---:|---:|
| 1x64 | 26.5 | 43.2 | 1.63x | 47.0 GB |
| 1x128 | 35.5 | 67.8 | 1.91x | 50.6 GB |
| 1x256 | 41.2 | **92.0** | **2.23x** | 57.5 GB |
| 2x256 | **OOM** | **149.5** | — | 76.5 GB |
| 4x256 | OOM | OOM | | |

Best throughput 41.2 -> 149.5 tok/s, **3.63x**: 2x256 did not fit before,
because the per-step scan's `states` tensor is 808 MB per layer at T=256 and
chunking keeps t/16 of them.

`gdn_chunk_core` collapsed from 40 lines to 8 by calling the same
`_gdn_chunk_fwd`: forward and backward share ONE implementation of the algebra,
so there is no old path left beside the new one.

## Precision, which is the whole point

Two properties, both gated:

- **The algebra is exact.** `_gdn_chunk_bwd` is the true adjoint of
  `_gdn_chunk_fwd`: in f64 it matches autograd to **< 1e-12**, term by term
  (`test_gdn_chunk_adjoint_is_exact`). A 25-term hand derivation needs this, and
  it caught nothing only because it was written against the oracle from the
  start.
- **The f32 error stays at the serial scan's level.** Same algebra, different
  reduction order, so f32 rounding differs. Measured against autograd on the
  SERIAL forward, 3 shapes x 3 seeds, worst relative error:

| chunk | worst rel | vs serial (3-9e-7) |
|---:|---|---|
| **16** | 4-12e-7 | **1.3-2.2x** |
| 32 | 1.1-2.3e-6 | 3-5x |
| 64 | 1.9-4.9e-6 | 5-10x |

**16, not the upstream 64.** The extra 4x of launches that 64 would have saved
is not worth a 5-10x error, and the error is not concentrated anywhere — it is
uniform across all eleven gradients (one, `dv`, is better than the serial one),
which is what a reduction-order difference looks like rather than a defect.

The old `test_gdn_bwd` gradchecks at T=3, which never leaves the first chunk;
`test_gdn_bwd_spans_chunks` runs at T=16 and T=37 so a partial tail is covered.

## What did not work

The f32 gap looked like the triangular solve `M = (I+L)^-1`, so I ran it in f64
(cheap: the matrices are 16x16). **No effect** — 6.92e-7 either way. Reverted.
The second time this session that a plausible mechanism cost a build before the
measurement that would have ruled it out.

## Rule

Prove the algebra separately from the arithmetic. In f64 the two forms agree to
1e-15, which turns "is the chunked backward correct?" into two smaller questions
with different answers: the derivation is exact, and the chunk length is a
precision knob. Without the f64 check, the 1.3-2.2x f32 gap is indistinguishable
from a wrong term.
