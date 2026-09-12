# V100 sm70 measured floors: 778 GB/s HBM, 88.8 TFLOP/s f16 — 2026-09-11

> Status: **measured on the V100**, two rows appended to the bench ledger;
> `bench --kernels` roofline on the 27B sm70 cell follows in the same entry.
> sm70 calibrates an **f16** tensor peak, not bf16 — Volta has no bf16 MMA.

## Context

The kernel roofline divides declared bytes/flops by a MEASURED floor, never a
datasheet ([calibration.py](../../src/tilerl/calibration.py)). Only the H20
sm90 had floors (3292 GB/s, 136 TFLOP/s bf16). ckl named the V100 a target
for the long-context work, so its two floors had to be measured before any
`%bound` column could render there.

## Two code facts the card surfaced

**sm70 has no bf16 tensor path.** `measure_bf16_peak_tflops` runs an
8192x8192 bf16 GEMM; on sm70 that has no tensor core to land on. The
calibration now measures `f16_peak_tflops` for pre-Ampere arches and the
roofline resolves the peak as bf16-where-present else f16
(`calibration()` returns the metric it used). The ledger section carries both
keys, the absent one `null` — a V100 row reads f16 88.8, an H20 row keeps
bf16 and ignores any f16 row. The row's `target` is the measured arch
(sm70/sm90 from capability), not a hardcoded sm90.

**A full 27B layer packed on-GPU OOMs a 32 GB card.** The kernel-roofline
fixture packs a random weight through `pack_fp4`, whose nearest-grid LUT
materializes ~8x the weight: a 17408x5120 layer asked for 37.9 GiB. That
fit the 95 GB H20 where the fixture was written, so nothing caught it. Packing
is untimed prep; `_pack_fp4_chunked` now packs and renorms in 2048-row chunks
on the weight's device — slices never leave the GPU, and pack/renorm are
per-row so the chunked result is bit-identical while the transient is bounded.

The same run also fixed the `%bound` column: it divided the COUNT-scaled bound
(all 48 GDN layers) by ONE call's ms, printing impossible 553-867%. The
column is the per-row bound over the per-call ms.

## Measured floors

Tesla V100-SXM2-32GB, warm CUDA-event medians (>=1 GiB D2D copy read+write;
8192 fp16 square GEMM), `tilerl bench --calibrate --card 0`:

| metric | value | datasheet ceiling |
|---|---:|---:|
| hbm_bw_gbs | **778.2** GB/s | ~900 |
| f16_peak_tflops | **88.8** TFLOP/s | 125 (fp16 tensor peak) |

The f16 number sits at 71% of the 125 TFLOP/s datasheet tensor peak — the
sustained, shareable floor the roofline should divide by, not the spec
maximum, consistent with the H20 measuring 136 vs its nominal ~148.

RESIDENCY measured: 27B resident peak **28.58 GiB = 28.33 static + 0.25
transient** on the V100, 4.9 GiB free — dense long-context cannot exceed ~32k
f32 there; 128k/256k on the V100 are sparse/cold-tier by construction. Decode
B=1/8 roofline captured in the same run (GEMVs 0.23-0.43 ms/call, 7-22% bound).

## 27B residency (serve --dry-run --record-residency)

| metric | value |
|---|---:|
| device_resident_bytes peak | **28,575,371,264** (26.61 GiB) |
| static | 28,326,046,756 (26.38 GiB) |
| transient | 249,324,508 (0.23 GiB) |
| free of 32 GiB | ~4.9 GiB |

## 27B prefill roofline (S=4096, B=1, fp8 KV)

`bench --kernels --model qwen38-27b --checkpoint … --prefill 4096`, measured
ms of the real registry GEMM on a single random weight, floors the V100 rows
above. Every %bound is now in (0,100]:

| kernel | count | ms/call | bound/call | %bound |
|---|---:|---:|---:|---:|
| in_proj_qkv (GDN) | 48 | 39.8 | 4.84 | 12.2% |
| gate/up_proj (GDN) | 64 | 65.0 | 10.97 | 12.6% |
| down_proj (GDN) | 64 | 60.0 | 10.97 | 13.7% |
| q_proj (attn) | 16 | 47.0 | 5.81 | 12.3% |
| o_proj (attn) | 16 | 22.7 | 2.90 | 12.8% |
| k/v_proj (attn, narrow) | 16 | 18.1 | 0.48 | 2.7% |
| lm_head (decode GEMV M=1) | 1 | 1.17 | 0.92 | 78.9% |
| paged_attention_prefill | 16 | pending | 37.2 | — |
| gdn_chunk_forward | 48 | pending | 52.7 | — |
| rmsnorm | 129 | pending | 0.11 | — |
| silu_mul | 64 | pending | 0.55 | — |

The big prefill GEMMs sit at ~12-14% of the f16/HBM roofline: sm70 reaches
about an eighth of either ceiling at M=4096 with the rung GEMV ladder — a
real headroom statement, but the prefill tick is GDN/attention-dominated
(their fused kernels have no linear timing fixture and render pending).
lm_head at M=1 is the one tight GEMV: 79% bound.

## Rule

A roofline floor is per-arch-specific in dtype as well as number: sm70's only
tensor peak is f16, and a bf16 GEMM there times the wrong (CUDA-core) path.
A fixture sized for a 95 GB card is a latent OOM on a 32 GB card — when an
untimed prep step builds an oversized scratch, chunk it on-device rather than
round-trip the weight through host.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-11 | V100-SXM2-32GB | cuda sm70 | HBM 778.2 GB/s, f16 peak 88.8 TFLOP/s appended |
| 2026-09-11 | V100-SXM2-32GB | cuda sm70 | 27B residency peak 28.58 GiB (28.33 static + 0.25 transient), 4.9 free |
| 2026-09-11 | V100-SXM2-32GB | cuda sm70 | dense 32k prefill 604.0 s (18.43 ms/tok, eager), decode tok/s pending |
