# Per-kernel roofline ledger for one 27B decode tick — CPU, 2026-09-10

> Status: pending-remote. The byte/flop declarations and `tilerl bench --kernels`
> table ship now; the measured ms / %-of-bound columns and the per-card
> calibration row land when the cards return from the aupai V4.1 block.

## Context

The band claims in wins/errors entries (decode bandwidth-bound, "22.8 GB per
tick", %-of-HBM) were prose: each bench/probe/profile script carried its own
byte product and a reader could not reproduce the number from the model. The cost
model (docs/design-cost-model.md) replaces that with one primitive —
`precision.nbytes(fmt, shape)` — and one declaration per launched kernel. This
entry records the derived 27B decode-tick table the kernel-cost unit prices.

## What worked

`tilerl bench --kernels --model qwen38-27b --batches 1,8` prints, per kernel at
a concrete decode shape: count over the tick, bytes moved, flops, and pending
ms/% columns. Weight bytes are enumerated from `param_specs` and priced with
nbytes(nvfp4) — no matrix dim is restated. The fp8 KV plane's scale group is the
model's head_dim (`kv_format(head_dim)`): one f32 per plane x head x token.

Derived totals (s=4096, fp8 KV, nvfp4 weights):

| batch | bytes / tick | flops / tick |
|---|---:|---:|
| B=1 | 14.31 GB | 0.070 T |
| B=8 | 18.61 GB | 0.557 T |

The B=1 tick is overwhelmingly the nvfp4 weight stream (13.7 GB of the 14.3 GB),
i.e. bandwidth-bound — the physics the roadmap states — with attention KV and
the GDN recurrence growing with B and s.

## Rule

A byte or flop number about a tick is derived through nbytes from the model
config and the kernel's declared shape, never a hand product in a probe script;
the attention-decode byte gate (tests/test_kernel_cost.py) pins the declaration
to what the KV pool actually allocates. Measured roofline waits for a card
calibration row (a copy kernel + a large GEMM), never a datasheet number.

## Results

| date | commit | machine | target | model | bytes B=1 | bytes B=8 | flops B=8 |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-10 | pending | CPU (derived) | cpu | qwen38-27b | 14.31 GB | 18.61 GB | 0.557 T |

Raw artifact: `tilerl bench --kernels --model qwen38-27b --batches 1,8`
(measured columns print `pending-remote`).
