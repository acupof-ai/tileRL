# ISO-Merger: two specialists merged without data beat the average — tiny/CPU, 2026-09-02

> Status: CPU-verified; 27B vs TIES/DARE `pending-remote`

## Context

`docs/design-rl-stack.md` §1: RLVR moves a linear's singular frames and leaves
its spectrum alone, so specialists that share a base can be composed offline
from their checkpoints — no rollouts, no data. `src/tilerl/merge.py` is that
procedure (arXiv 2607.19331, Appendix E) and `tilerl merge` is the CLI. Per 2D
weight: SVD the base, SVD each specialist and sign-align its singular vectors
to the base, take the frame displacement, project it onto the Stiefel tangent
space, drop the trailing 10% of modes, solve one ridge Gram system for the
unit-retention coefficients, sum, re-project, polar-retract, rebuild
`U* Σ₀ V*ᵀ`. Non-matrix params are averaged.

Two deviations from the reference script, both deliberate: the ridge is
relative to the Gram's own scale (`ridge * mean(diag Γ)`, default 1e-3)
instead of an absolute 1e-12, so it means the same thing for a 64-wide tiny
weight and a 5120-wide 27B one; and the CLI loads every checkpoint as bf16
masters in RAM (`# ponytail:` stream per shard for the 27B).

## What Worked

Tiny model, `RefBackend`, two SFT specialists: 15 AdamW steps at lr 1e-3, one
on fixed random batch A, one on batch B (`tests/test_merge.py`). Losses are
`train_step`'s causal CE:

| model | loss on A | loss on B |
|---|---:|---:|
| base | 22.335 | 21.990 |
| specialist A | 14.945 | 21.941 |
| specialist B | 22.291 | 13.565 |
| average merge (`W₀ + mean ΔW`) | 18.460 | 17.550 |
| **ISO merge** | **15.995** | **14.726** |

ISO recovers 87% of specialist A's gain on A and 86% of B's gain on B; the
average recovers 52% and 53%. Each specialist alone does nothing for the other
batch, so the merged model is doing both jobs from one set of weights.

Procedure gates, float32 params so the container's rounding is not measured:

- K=1 returns the specialist: max relative Frobenius error over all 2D weights
  **3.3e-3** (max displacement 1.2e-2). Not exact by construction — the trailing
  10% of modes are masked and the retraction is first order — the gate is 1e-2.
- Spectrum kept: max relative singular-value error vs the base **8.9e-7**
  (gate rtol 1e-3).

Whole gate: 1.3 s on CPU.

## Rule

Frame-space merging is the default merge for specialists that share a base:
it keeps most of each specialist's own gain where averaging keeps half. The
27B comparison against TIES/DARE on our own RL specialists is the open claim —
a tiny-model SFT result licenses the procedure, not the 1.6-pt paper number.

## Results

| date | commit | machine | target | model | K=1 rel err | spectrum rel err | loss A base/avg/iso | loss B base/avg/iso |
|---|---|---|---|---|---:|---:|---|---|
| 2026-09-02 | this | Mac (no GPU) | cpu | tiny | 3.3e-3 | 8.9e-7 | 22.34 / 18.46 / 16.00 | 21.99 / 17.55 / 14.73 |
| pending-remote | | pod | cuda | qwen38-27b | | | TIES / DARE / ISO on the real task | |

Raw artifacts: `TILERL_TARGET=cpu uv run pytest -q -s tests/test_merge.py`.
