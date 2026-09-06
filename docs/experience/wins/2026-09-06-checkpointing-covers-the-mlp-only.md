# Checkpointing covers the MLP only, so every layer's attention and GDN activations stay live — H20 sm90, 2026-09-06

> Status: Shipped (measurement) · the fix is the next PR

## Context

A GRPO step at gen 4096, group 8, OOMs on one 95.22 GiB card, and the failing
allocation is an MLP intermediate rather than the attention score matrix that was
blamed for it
([errors entry](../errors/2026-09-06-chunked-attention-was-the-wrong-term.md)).
That left the real question unanswered: **what holds the memory during the
forward?** The OOM frame sits inside `autograd.checkpoint`, which exists precisely
so a segment's activations do not coexist with the rest of the forward — so
something is accumulating that checkpointing does not cover.

## What Worked

`scripts/prof_forward_memory.py`: one forward, `memory_allocated` read at every
checkpoint boundary, plus a storage-deduped census of live cuda tensors by shape.

**`model.py:434` is the only `autograd.checkpoint` call in the model, and it wraps
`_mlp_body`.** `_full_attn` and `_gdn` run unwrapped at `model.py:477-480`, so
every layer's attention and GDN activations stay on the tape for the whole
forward. That is the finding: checkpointing covers roughly a third of each layer.

Measured at gen 4096 (T=4352), 64 of 64 segments recorded:

| point | torch-allocated | device used (`mem_get_info`) |
|---|---:|---:|
| before the forward | 0.070 GiB | 0.443 of 95.223 GiB |
| segment 1 end | 1.810 GiB | |
| segment 64 end | 44.738 GiB | |
| forward end / process peak | **54.038 GiB** | 60.450 of 95.223 GiB |

Two runs of the same probe, 54.035 and 54.038 GiB — **0.007% apart**.

**+42.929 GiB accumulated across 64 segments — 697.8 MiB per segment against
85.0 MiB for one retained `[T,hidden]` input.** So `checkpoint`'s retained
`args = (layer_idx, x, kv, backend)` account for **12%** of the accumulation
(5.3 GiB of 42.9); the other 88% is activations the checkpoint never wrapped.

### Live at forward end — per-shape TOTALS, not single tensors

53.984 GiB across 36 shape classes. Each row sums every live tensor of that
shape; the copy count is the row's bytes divided by one tensor's.

| shape | GiB | copies | what it is |
|---|---:|---:|---|
| `(4352, 16384)` | **12.484** | 47 | GDN fused `qkvz` output — one per GDN layer |
| `(1, 4352, 5120)` | 10.625 | 128 | `[1,T,hidden]` — 2 × all 64 layers |
| `(4352, 5120)` | 8.114 | 98 | `[T,hidden]` — 2 × the GDN count |
| `(4352, 248320)` | 8.052 | 2 | logits, **4.026 GiB each** |
| `(4352, 14336)` | 3.719 | 16 | one per full-attn layer |
| `(1, 4352, 24, 256)` | 3.188 | 32 | 2 per full-attn layer, 24 heads × 256 |
| `(208896, 128)` | 2.391 | 24 | `[T×48, 128]` GDN per-head |
| `(104448, 256)` | 1.594 | 16 | `[T×24, 256]`, one per full-attn layer |
| `(1, 4352, 6144)` | 1.594 | 16 | GDN value, 48 × 128 |
| `(248320, 5120)` | 1.184 | 0.2 | **not an activation** — the lm_head weight, under 4 B/element |

**Every activation row's copy count is a layer count**: 47 and 24 against the 48
GDN layers, 16 and 32 against the 16 full-attn layers, 128 = 2 × 64. That
self-consistency is the census's own check, and the 0.2-copy row is how a weight
distinguishes itself from an activation in the same table.

`16384` is not a config field and was not guessed: `linear_q_dim 2048 +
linear_k_dim 2048 + linear_v_dim 6144 + z 6144 = 16384`, the fused projection
built at `model.py:60-61`.

Grouped by what they belong to: `[T,hidden]` **18.739 GiB (34.7%)**, GDN-shaped
**16.614 GiB (30.8%)**, full-attn-shaped **8.767 GiB (16.2%)**, logits
**8.052 GiB (14.9%) in two tensors**.

## Controls

| control | reading |
|---|---|
| tape actually recording | 64 checkpoint entries on the tape for 64 layers, asserted before any number is printed |
| the same assertion on a call count | **does not discriminate** — `checkpoint` is called either way and returns early at `autograd.py:41`; measured on tiny: 2 calls in both arms, 2 tape entries with `recompute` and **0** without |
| census not double-counting views | storages deduped by `(data_ptr, nbytes)`; every copy count lands on a layer count (47, 24, 16, 32, 128) rather than an arbitrary figure |
| the whole measurement repeated | 54.035 then 54.038 GiB peak, 42.925 then 42.929 GiB accumulated — two runs, 0.007% and 0.009% apart |

## Not established

- **`memory_allocated` is quotable only as a delta, now confirmed rather than
  suspected.** It reads 0.070 GiB before a 27B forward while the driver reports
  **0.443 GiB** used; at forward end it reads 54.038 against the driver's
  **60.450**. So the weights and ~6.4 GiB of other device memory live outside
  torch's allocator, and no figure of the form "X above resident" is meaningful.
  The 53.968 GiB forward delta and the per-segment rises are differences within
  one measurement and stand.
- **Forward only, gen 4096, batch 1.** No backward, so this is not the step peak,
  and the OOM happens with a backward in flight.
- **Nothing here says the accumulation is avoidable**, only that checkpointing as
  currently placed does not avoid it.

## Rule

A checkpoint bounds only what its own callable computes. `autograd.checkpoint`
wraps `_mlp_body`, so reasoning of the form "activations are checkpointed, they
cannot accumulate" is true for the MLP and false for the other two thirds of the
layer — 697.8 MiB per segment against the 85.0 MiB the wrapper retains. Before
trusting a memory-bounding mechanism, read what it is wrapped around, and measure
the rise per unit rather than the size of the thing you expect to see.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-06 | 61b9e43 | H20 card 6 | cuda sm90 | 27B, gen 4096 | n/a | n/a | forward peak 54.038 GiB |

Raw artifacts: `/work/fwdmem.log` (probe `87a0b4a1aaca`) and `/work/fwdmem2.log`
(probe `e6d17462707b`, the instrument-corrected re-run these figures come from),
both against engine tree `61b9e43` read in the launch call.
