# TILERL_RED_TILE=64 made the weight gradient exactly zero

## Context

`_RED_TILE` is a documented A/B lever (`TILERL_RED_TILE`, added by
`errors/2026-08-29-mma8-is-register-bound.md`). A code review read the source
and predicted that setting it to 64 would silently zero the weight gradient.
Nobody had run that configuration. Reproduced on sm90 before anything was
changed.

## Root Cause

Four MMA kernels loop over their reduction dimension as `X // _RED_TILE`
(`kernels_linear.py:41, 68, 93, 1009`) — floor division, so a reduction dim
that is not a multiple of the tile drops the tail, and one smaller than the
tile makes the loop zero-trip and returns the cleared accumulator.

`backend.py:57` defines `_MMA_RED = kernels_linear._RED_TILE` with a comment
saying the backend imports the name to pad, so it is defined once. It was used
at exactly one call site (`linear_frozen_bwd`). The six pads in `linear` and
`linear_bwd` hardcoded `32`.

Measured at `TILERL_RED_TILE=64`, H20:

| shape | before | after |
|---|---|---|
| `gw`, M=32 | **all zeros** (rel err 1.0000) | rel err 0.0000 |
| `gw`, M=275 (padded to 288, `288 // 64 = 4`) | rel err **0.3138** — 32 of 288 reduction rows dropped | rel err 0.0000 |
| `gx` | rel err 0.0000 (its reduction is a model dim, already a multiple of 64) | 0.0000 |
| tiny model, one tape step | **15 of 27 parameter gradients exactly zero** | 0 of 27 |

At M=32 training would have reported a finite loss and stopped learning with
no error and no warning.

## Fix

The six pads route through `_MMA_RED`
(`backend.py:283-284, 302-303, 311-312`), so the padding and the kernel's step
are the same number by construction. The fp4 plan's K pads are constants rather
than derived, so `kernels_linear.py` now also refuses a tile that does not
divide 256 — the smallest of them.

Two gates. `tests/test_ops_parity.py::test_reduction_pads_are_the_reduction_tile`
fails if a literal comes back or if a plan K pad stops being a multiple of the
tile, and runs on any target. `test_production_model_gradcheck` catches the
gradient itself, but only on sm90 with the lever set — which is worth writing
down: **the gate that would have caught this exists and has never run in the
configuration that breaks.** CI is CPU-only, and the CPU and Metal cells use
`kernels.make_gemm_*`, which never read `_RED_TILE`.

## Rule

A tuning lever that changes a loop bound must feed every pad that loop reads,
from one name. Same shape as the two other defects this session: a loop bound
that floor-divides to zero and hands back its untouched buffer. When a review
predicts one, run the failing configuration before fixing it — the numbers
above are why the fix is trustworthy.
