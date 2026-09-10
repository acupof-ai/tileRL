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
The served checkpoint mixes faces: 168 on-disk block-16 nvfp4 linears, 233 fp8
linears, and 96 `in_proj_a`/`in_proj_b` weights that ship bf16 and are
**repacked at load with pack_fp4's block 32**, not 16.

## What worked

`model.checkpoint_weight_faces(cfg, dir) -> dict[param_key, (shape, Format)]`
classifies every served weight's device face from
`precision.checkpoint_weight_specs` (headers only), mapping HF names through
`_param_key_for` exactly as `load_hf` (vision/MTP tensors fall out), and
applies the loader's two transforms:

1. a bf16 linear in `fp4_param_keys` is reported `nvfp4_dev_b32` under
   `cfg.fp4` — `pack_fp4` defaults to block 32, so the served scale grid is
   f32 per 32 along K plus one f32 per row, wider blocks than on-disk NVFP4's
   16 (`nvfp4_dev`);
2. a 3-D disk conv1d `[C,1,K]` is flattened to `[C,K]` like load_hf's reshape.

The map is exhaustive over `param_specs` (851 keys on the 27B), so its nbytes
sum is the resident total to the integer. `TickShape.faces` carries it; one
`_linear_launches` enumeration feeds both `tick_rows` (decode) and
`prefill_rows` (#463), collapsing equal `(key, shape, face)` GEMMs to one
counted row. `tilerl bench --kernels --checkpoint DIR` (and `--prefill S
--checkpoint`) prints the real population with a `face` column; without the
flag both tables are unchanged (all-`nvfp4` config face).

Derived totals, s=4096, fp8 KV, checkpoint device faces:

| batch | old (all-fp4) | checkpoint faces | delta | flops |
|---|---:|---:|---:|---:|
| B=1 | 14.88 GB | **22.36 GB** (22,359,395,456 B) | +7.477 GB | 0.072 T |
| B=8 | 18.16 GB | **25.63 GB** (25,634,411,648 B) | +7.477 GB | 0.577 T |

The delta is identical at B=1 and B=8 because weights stream once per tick;
only KV/state/activations grow with batch. Per-family deltas
(weight+activations over launches):

| family | all-fp4 | checkpoint | delta |
|---|---:|---:|---:|
| attn q/k/v/o (16 full layers) | 0.945 GB | 1.681 GB | +0.736 GB |
| GDN in_proj qkv/z/a/b + out_proj (48) | 3.132 GB | 5.561 GB | +2.429 GB |
| MLP gate/up/down (56 nvfp4 + 8 fp8 layers) | 9.635 GB | 13.389 GB | +3.754 GB |
| lm_head | 0.716 GB | 1.273 GB | +0.558 GB |
| **total weight delta** | | | **+7.477 GB** |

Prefill S=1024 prices 56.99 GB against 49.51 GB all-fp4 — the same weight
delta streamed once; flops are unchanged (50.51 T).

## Byte-exact oracle (no tolerance)

A band sized to a headline's rounding hid two prior defects (~4 MB and
2.95 MB). The gates now assert exact integers:

- `tests/test_precision.py::test_27b_header_weight_bytes_match_the_recorded_exact_integer`
  — raw header sum **26,240,262,816 B** (includes 1.77 GB vision/MTP), `==`.
- `tests/test_kernel_cost.py::test_27b_served_weight_faces_equal_load_hf_resident_exact`
  — served-face sum **24,436,981,888 B**, `==` a live `load_hf` model's
  resident tensor storage (1845 tensors), and `set(faces) == set(param_specs)`.
- `tests/test_kernel_cost.py::test_bf16_linear_is_reported_as_the_face_load_hf_serves_under_fp4`
  — a packed key's `nbytes(nvfp4_dev_b32)` equals pack_fp4+renorm storage,
  and is asserted `!=` the block-16 face, so a block regression is red.

Weight-only decode stream (GEMV activation terms stripped) is **21.889 GB**;
the 2026-09-03 loader measurement was 21.89 GB streamed / 24.44 GB resident.

## Rule

A per-tick byte number prices the face the kernel is actually served, read from
the checkpoint headers — and the loader's load-time transforms (bf16→fp4
packing, its block size, conv1d reshape) are part of the served face. A
config-derived format population is a hypothesis; the 27B's is mixed and the
delta is a third of the tick. A per-tensor formula gets a byte-exact oracle,
never a band that rounds the answer into agreement.

## Results

| date | machine | target | model | bytes B=1 | bytes B=8 | flops B=8 |
|---|---|---|---|---:|---:|---:|
| 2026-09-11 | pod (derived, header-only) | cpu | qwen38-27b @ Qwen3.8-27B-NVFP4 | 22.36 GB | 25.63 GB | 0.577 T |

Raw artifact:
`tilerl bench --kernels --model qwen38-27b --batches 1,8 --checkpoint /work/Qwen3.8-27B-NVFP4`
(measured columns print `pending-remote`). Supersedes the 14.88/18.16 GB rows of
[2026-09-10-kernel-roofline-ledger.md](2026-09-10-kernel-roofline-ledger.md),
which stay valid for the config-face-only question.
