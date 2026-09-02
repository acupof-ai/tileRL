# ISO fixed-spectrum optimizer on the tape — cpu, 2026-09-02

> Status: pending-remote (27B); tiny model shipped

## Context

ISO (arXiv 2607.19331) keeps every 2D weight's singular values fixed at their
initial values and trains only the singular frames: `W = U S V^T`, the base
optimizer moves `U` and `V`, a polar retraction puts them back on the Stiefel
manifold, and `W` is rebuilt. The paper's claim is a better inductive bias for
RLVR post-training at ~7% extra step time. `src/tilerl/iso.py` (93 lines)
wraps `Adafactor` (or `AdamW`) behind the same `streams` / `begin` /
`step_one` contract, so `train._step`'s streamed full-parameter path is
unchanged and every gradient is still freed the moment it is applied.

Frame gradient is the chain rule, `G_U = G V S`, `G_V = G^T U S` (paper
eq. 34; no tangent projection). Retraction is 5 Newton-Schulz iterations,
`X <- 1.5 X - 0.5 X (X^T X)`, matmuls only — the SVD runs once, at first
sight of the weight. Frames are fp32 by default (`frame_dtype`).

## What Worked

Gates in `tests/test_iso.py`, all CPU, 4 s total:

- frame gradient vs central finite difference along a random Stiefel tangent
  through the retraction, fp64: relative error **1.4e-10**.
- after 3 steps on a random 64x48 bf16 weight (Adafactor and AdamW bases):
  `max|U^T U - I|` = **4.2e-7**, same for `V`.
- after 5 `train_step`s on the tiny model: singular values of the fp32 frame
  product move **5.9e-6** relative; the bf16 param's spectrum sits **4.4e-4**
  (relative to `s_max`) off its start, which is bf16 rounding, not drift.
- 8 `train_step`s on one fixed 2x32 batch, `ISO(Adafactor(lr=1e-2))`:

  | step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | ms/step |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | ISO | 24.78 | 14.23 | 7.57 | 3.62 | 1.59 | 0.62 | 0.19 | 0.025 | 11 |
  | Adafactor | 24.78 | 14.28 | 7.70 | 3.60 | 1.60 | 0.57 | 0.10 | 0.029 | 7 |

  Same curve as the unconstrained optimizer on the tiny model; the 4 ms is
  two extra matmuls per frame and the rebuild, on 26 tiny matrices.

CLI: `tilerl train --optim iso` on the full-parameter SFT path. Every 2D
tensor is treated as a frame pair, `conv1d (96x4)` included — with q = 4 its
spectrum is 4 numbers, harmless, and one rule beats a name list.

Ceilings, stated:

- **Not wired into `--rl` / `--opd`.** Those train LoRA adapters on a frozen
  base; ISO has no LoRA variant. Full-parameter RL would also need per-step
  re-quantization of the updated master into the served fp4 bytes, which does
  not exist yet.
- **Memory on the 27B.** `U`, `S`, `V` in fp32 are two fp32 copies of every
  2D weight — 4x the bf16 master, ~200 GiB — plus Adafactor's factored state
  on each frame. `frame_dtype=torch.bfloat16` halves it at the cost of the
  orthonormality error above; not measured. Newton-Schulz on a 5120x5120
  frame is `pending-remote`.

## Rule

An optimizer that only changes what a step does to one tensor plugs in
behind `begin` / `step_one` and nothing in `train.py` moves. Gate a
manifold optimizer on its invariant (spectrum, orthonormality) measured in
the dtype the invariant lives in — the bf16 param is a rounding of it.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-02 | this | Mac (no GPU) | cpu | tiny | n/a | n/a | 11 ms/train step, 2x32 |
| pending-remote | | pod | cuda | qwen38-27b | | | |

Raw artifacts: `tests/test_iso.py` (`python tests/test_iso.py` prints the loss curve and s/step).
