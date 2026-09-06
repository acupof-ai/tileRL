# A cp-only mesh inherited the world as TP, so the residual all-reduced across CP ranks — cpu, 2026-09-06

> Status: the `init_tp` half is FIXED here; the `_gdn_cp` half ships with the
> GDN-under-CP branch, which is where that function lives. End to end the two
> together take `tests/cp_model_world2.py` from logits rel **1.2e+00 to 8.8e-07** at
> cp=2. Both numbers are **CPU-arm**; see [Not established](#not-established) for the
> sm90 case, which has not been run.

## What is in this change

The `init_tp` fix in both copies, and one assertion in `tests/cp_world2.py`. The
second defect below is in `_gdn_cp`, a function that does not exist on `main` — it
arrives with the GDN-under-CP branch and is fixed there. It is described here
because the four-cell measurement is what separates the two, and neither row means
anything alone.

## Context

`feat/cp-gdn` wires the gated-delta layer under context parallelism. The op-level
gates were green — `cp_world2.py` (out/dQ/dK/dV to 4.8e-07), `gdn_cp_scan.py` (a
chunk started from its composed prefix matches the sequential scan to 3.0e-07) —
and the end-to-end model gate was **rel 1.2e+00**, i.e. the output was wrong by
its own magnitude. Parked with the failure at "layer 0 add".

## Root cause: two independent defects, and one was not in the CP code at all

`Backend.init_tp` (`packages/tilerl-kernels/.../backend.py:190`, and the same
lines in `src/tilerl/testing.py`) read:

```python
if tp_groups:
    ...
else:
    self.tp_world, self.tp_rank = world, rank
```

A cp-only run passes `cp_groups` and no `tp_groups`, so it took the `else` and
**set `tp_world = 2` with TP switched off**. `_add_via` then all-reduced the
`o_proj` residual across the two **CP** ranks: correct operands, corrupted sum.
Every tensor shape stayed local, so nothing looked wrong anywhere — the only
symptom was the number at the end.

The `else` was correct for its one original caller: the TP gates call the bare
`init_tp(world, rank)` with no axis named at all, and mean "the whole world is one
TP group". It became wrong when a second axis learned to own the world.

The second defect was in `_gdn_cp`: the composed prefix `a_pre[i] @ state.float()
+ b_pre[i]` was rounded back with `.to(state.dtype)` (bf16) before entering the
chunk. The sequential path does not round there, so CP diverged from it.

## Four cells, because two defects were live at once

`cp_model_world2.py`, cp=2, rank 0 logits rel, gate 1e-5:

| `init_tp` | prefix dtype | rel | rc |
|---|---|---|---:|
| inherits `world` | bf16 (as parked) | 1.2e+00 | 1 |
| inherits `world` | f32 | 1.2e+00 | 1 |
| `tp_world=1` | bf16 | 3.0e-05 | 1 |
| `tp_world=1` | f32 | **8.8e-07** | 0 |

Measured as four separate runs with `__pycache__` cleared between them, since a
restored file is not a restored import. Reading the rows: the all-reduce dominates
and hides the rounding entirely (rows 1 and 2 are identical), and fixing only the
all-reduce lands at 3.0e-05 — **3x over the gate**, which would have read as "CP
is nearly right, loosen the tolerance". The residual after both is bf16-scale
rounding, not a third defect.

## Controls

| reverted | goes red |
|---|---|
| the `init_tp` fix | `cp_world2.py` — `AssertionError: cp-only mesh set tp_world=2: the residual add will all-reduce across the CP ranks` |
| the `init_tp` fix | `cp_model_world2.py` at rel 1.2e+00 |
| the f32 prefix | `cp_model_world2.py` at rel 3.0e-05 |

The new assertion lives in `cp_world2.py` rather than only in the model gate
because that gate is an **op** gate — it calls `linear_attn_chunk` directly and
never reaches `_add_via`, so it could not have seen this defect by its own
numbers. The assertion is what makes it able to.

## Not established

- **The 1e-5 gate is a CPU-arm number.** The measured 8.8e-07 is the CPU
  reference arm's residual.
- **The sm90 WY case is pending-remote, not a number.** On sm90 the WY scan
  rounds the composed prefix on entry, same as the sequential path at every
  `_WY_CHUNK` boundary; the CP gate's 1e-5 has not been run there.
- Nothing here says whether other axis combinations are right. Only cp-only and
  the bare TP call were measured; `dp+cp`, `tp+cp` take the same new branch by
  construction but were not run through a model gate.

## Rule

**A probe that validates one arm is not a result about the mechanism.** Deciding
whether to keep the bf16 round, I probed the CPU arm — where `reference.
gdn_chunk_core` takes `_f32(state)`, so removing the round changes the value —
and was about to report it as general. sm90's `_gdn_wy_core` passes
`_bf16(state)` and carries a comment saying the gemm operand rounds on entry
either way, so the same probe there answers the opposite. Before reporting a
dtype or precision finding, name every arm the code path has and say which one
the number came from.
