# The decode tick streams 22.36 GB at B=1, not 14.88 — weights priced from the checkpoint — CPU/pod, 2026-09-11

> Status: derived (header-only, no GPU). The measured ms / %-of-bound columns
> stay pending-remote; these byte totals need only the safetensors headers and
> ran on the pod at `/work/Qwen3.8-27B-NVFP4` without a card.

## Context

The 2026-09-10 kernel ledger priced every linear at the all-nvfp4 config face
(`nbytes(nvfp4)` over `param_specs`), reporting 14.88 GB (B=1) and 18.16 GB
(B=8) per decode tick. The config says which keys *can* be fp4; it does not say
which weights the checkpoint shipped quantized — the same population trap named
in [errors/2026-09-03-fp4-param-keys-is-not-the-fp4-tensors.md](../errors/2026-09-03-fp4-param-keys-is-not-the-fp4-tensors.md).
The served checkpoint mixes faces: 264 nvfp4 linears and 233 fp8 linears, and
96 of the nvfp4-keys ship bf16 and are packed by `load_hf` at load time
(`model.py`, the `cfg.fp4` pack loop).

## What worked

`model.checkpoint_weight_faces(cfg, dir)` classifies every served linear's
device face from `precision.checkpoint_weight_specs` (headers only), mapping HF
names through `_param_key_for` exactly as `load_hf`, and applies the loader's
one transformation: a bf16 linear in `fp4_param_keys` is reported `nvfp4_dev`
under `cfg.fp4`, because pack_fp4 at load is what serves it. `TickShape.faces`
carries the map; `tick_rows` enumerates every layer's launches and collapses
equal `(key, shape, face)` GEMVs to one row, so the table shows the real
population (56 nvfp4 gate/up/down + 48×2 packed in_proj_b/a, all attention and
GDN qkv/out and lm_head fp8).

`tilerl bench --kernels --checkpoint <dir>` prices weights from that map;
without the flag the table is unchanged (all-`nvfp4` config face).

Derived totals, s=4096, fp8 KV, checkpoint device faces:

| batch | old (all-fp4) | checkpoint faces | Δ | flops |
|---|---:|---:|---:|---:|
| B=1 | 14.88 GB | **22.36 GB** (22,362,344,576 B) | **+7.48 GB** | 0.072 T |
| B=8 | 18.16 GB | **25.64 GB** (25,637,360,768 B) | **+7.48 GB** | 0.577 T |

The Δ is identical at B=1 and B=8 because weights stream once per tick; only
KV/state/activations grow with batch. Per-row family deltas (weight+activations, summed over launches, identical at
B=1 and B=8 because weights stream once):

| family | all-fp4 | checkpoint | Δ |
|---|---:|---:|---:|
| attn q/k/v/o (16 full layers) | 0.945 GB | 1.681 GB | +0.736 GB |
| GDN in_proj qkv/z/a/b + out_proj (48) | 3.132 GB | 5.564 GB | +2.432 GB |
| MLP gate/up/down (56 nvfp4 + 8 fp8 layers) | 9.635 GB | 13.389 GB | +3.754 GB |
| lm_head | 0.716 GB | 1.273 GB | +0.558 GB |
| **total weight Δ** | | | **+7.48 GB** |

The checkpoint's fp8 keys are all the per-channel face (`fp8_dev`, grid plus a
real f32/row); `load_hf`'s plain `.weight_scale` branch stores that row. The
96 in_proj_b/a weights ship bf16 and are packed at load, so their served face
is nvfp4_dev like the other 264; `checkpoint_weight_faces` applies that
loader transform rather than reporting the disk face.

Cross-checks against the 2026-09-03 loader measurement: the table's weight-only
stream (GEMV activation terms stripped) is **21.892 GB**, measured 21.89 GB;
and all served faces summed — including the embed gather, which the tick table
does not stream — give **24.440 GB** resident, measured 24.44 GB. Both
agreements are under 0.1% with no fitted terms.

## Rule

A per-tick byte number prices the face the kernel is actually served, read from
the checkpoint headers — and the loader's load-time transforms (bf16→fp4
packing) are part of the served face. A config-derived format population is a
hypothesis; the 27B's is mixed and the delta is half the tick (7.48 of 22.36 GB).

## Results

| date | machine | target | model | bytes B=1 | bytes B=8 | flops B=8 |
|---|---|---|---|---:|---:|---:|
| 2026-09-11 | pod (derived, header-only) | cpu | qwen38-27b @ Qwen3.8-27B-NVFP4 | 22.36 GB | 25.64 GB | 0.577 T |

Raw artifact:
`tilerl bench --kernels --model qwen38-27b --batches 1,8 --checkpoint /work/Qwen3.8-27B-NVFP4`
(measured columns print `pending-remote`). Supersedes the 14.88/18.16 GB rows of
[2026-09-10-kernel-roofline-ledger.md](2026-09-10-kernel-roofline-ledger.md),
which stay valid for the config-face-only question.
