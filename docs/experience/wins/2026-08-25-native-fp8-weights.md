# Native FP8 weights: GDN projections kept native, pure fp8 WGMMA prefill — H20, 2026-08-25

> Status: Shipped

## Context

The 27B checkpoint's GDN projections (`in_proj_qkv`/`in_proj_z`/`out_proj`) ship
as ModelOpt FP8-block (e4m3 weight + per-128-block scale). `load_hf` dequantized
them to bf16, and on `cfg.fp4=True` re-packed them to fp4 — so the prefill path
ran the fp4→e4m3 dequant + fp8 WGMMA kernel (`linear_fp4_fp8`), whose K-loop
dequant held it at ~21% of the 296 TFLOPS fp8 peak. This change keeps the
checkpoint's FP8 weights native and runs a pure fp8 WGMMA (no K-loop dequant).

The task's premise (GDN projections "dequantized to bf16, capping the prefill
roof at 3835 tok/s") was based on the SOTA-all-levers bench's byte analysis,
which the 2026-08-25 bf16-GEMV entry corrected: on `cfg.fp4=True` every
projection is fp4-packed, so the GDN ran the fp4 path, not bf16. The lever is
nevertheless real — the fp4 path's K-loop dequant is the bottleneck, and native
fp8 removes it.

## What Worked

**Loader** (`model.py:load_hf`): the two FP8 detection branches (ModelOpt
FP8-block `weight`+`weight_scale_inv`, per-tensor FP8 `weight`+`weight_scale`)
now keep the e4m3 weight in `<key>.w8` and the per-128-block scale in
`<key>.wscale` (a per-tensor scalar is expanded to the same
`[ceil(N/128), ceil(K/128)]` layout so one kernel serves both). The bf16
dequant is kept as the recording-only master (same convention as the fp4
masters — the STE grad lands on it). `cfg.fp4` packing skips native-fp8 keys:
re-packing would lose the e4m3 precision and force the K-loop dequant path.
`in_proj_a`/`in_proj_b` (N=48 < 128) ship bf16 in the checkpoint and stay
bf16 — the block format can't quantize them.

**Kernels** (`kernels_mma.py`):
- `make_linear_fp8_mma` (prefill, M>1): e4m3 weights + e4m3 activations,
  per-128-block weight scale applied to the accumulator per K-chunk
  (block_K=128 = fp8 WGMMA K=32 × 4 steps), per-token activation scale as one
  divide in the epilogue. SOTA copy: `examples/deepseek_deepgemm/
  example_deepgemm_fp8_2xAcc.py` (the `C_local_accum += C_local * scale_b`
  per-chunk pattern). No K-loop dequant — the loop body is copy+copy+gemm+scale.
- `make_linear_fp8_gemv` (decode, M=1): the bf16 GEMV schedule with e4m3 W
  (micro_size_k=16, 1 byte/elem vs the bf16 master's 2) and one per-128-block
  scale lookup per chunk (block_K=512 = 4 scale blocks; the 16-elem slice never
  crosses a block boundary). bf16 X (no activation quant, unlike the MMA path).

**Wire**: `Model._linear` dispatches `.w8` linears to `backend.linear_fp8`
(sm90 M>1 → MMA kernel, M=1 → GEMV kernel, CPU/metal → bf16 master through the
floor). `autograd._linear_fp8` STE handler (master kwarg, same as `_linear_fp4`).
`reference.dequant_fp8`/`linear_fp8`/`linear_fp8_bwd` replace `dequant_fp8_block`
(one layout, one decoder).

**Parity**: `tests/test_ops_parity.py` — `linear_fp8_parity` (CUDA: identical-
quant reference, same per-token e4m3 activation quant; the weight side is exact
since it's native e4m3), `linear_fp8_gemv_parity` (f32 dequant reference — the
GEMV's bf16 X rounding is the only error), `linear_fp8_bwd` (STE). Local CPU
72 passed; pod CUDA sm90 31 passed (allclose rtol=1e-2).

**Prefill bench** (slice4 = 3 GDN + 1 full-attn, the 27B's exact 3:1 mix;
isolated GDN bench `scripts/bench_fp8_gdn.py`, same-process back-to-back so the
ratio is contention-independent, M=512):

| path | GDN 9 linears | TFLOPS | % of 296 peak |
|---|---:|---:|---:|
| fp4 (pack the master, `linear_fp4_fp8`) | 5.657 ms | 62.6 | 21% |
| native fp8 (`linear_fp8`) | 3.819 ms | 92.8 | 31% |

**Speedup 1.48x** on the GDN projections. Best single linear: 145.7 TFLOPS
(49% of peak). The fp8 path's per-chunk accumulator scaling (deepgemm 2xAcc)
costs some pipeline efficiency vs a pure WGMMA, but removes the dequant
entirely — net 1.48x.

The total-prefill win is bounded by the GDN fraction: the GDN projections are
~23% of the slice's projection FLOPs; the MLP (NVFP4, unchanged fp4 path) is
~77%. Slice4 prefill-512 before (idle GPU): 32.79 ms GPU sum → 15117 tok/s
slice, 978 tok/s extrapolated. Replacing the GDN's 5.66 ms with 3.82 ms gives
~30.9 ms → ~16.0k tok/s slice, ~1.06k extrapolated (~9% total). The 3800
tok/s target needs the MLP on a faster path too — but the MLP ships NVFP4
(fp4), not FP8, so native retention doesn't apply there; its lever is fp4
dequant efficiency, a separate piece of work.

**Decode** (slice4, graph-captured, back-to-back same phase): the fp8 GEMV
keeps the native fp8 weights on the decode path (1 byte/elem) instead of
falling back to the bf16 master GEMV (2 bytes/elem — a 3.2x byte regression
on the GDN projections). It is also slightly faster than the fp4 GEMV it
replaces: 2.672 ms/tick (374.2 tok/s) vs 2.799 ms (357.2 tok/s) — the fp8
GEMV has no e2m1 dequant stage, and the extra bytes (1.65x fp4) are cheaper
than the dequant compute. The bf16-master fallback would have read 3.2x the
bytes; the fp8 GEMV is the right decode path for native fp8 weights.

## Rule

Keeping a checkpoint's native quant format is a loader change, not a kernel
change: store the packed weight + scale as param siblings, keep the bf16
dequant as the recording-only master, and let the sm90 dispatch pick the native
kernel. And on a shared GPU, the only contention-independent kernel comparison
is a same-process back-to-back ratio — absolute CUDA-event numbers shift with
the co-tenant's phase (the profile_slice before/after pair moved 15117 → 5009
tok/s between an idle and a contended window, a phase change, not a
regression).

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | a536ff8 | H20 idle | cuda/sm90 | 27B slice (3 GDN + 1 FA) | 0.0640 (512-tok) | 2.799 (graph) | 15117 prefill / 357 decode |
| 2026-08-25 | c59da8b | H20 | cuda/sm90 | 27B slice (3 GDN + 1 FA) | — | 2.672 (graph) | 374 decode |
| 2026-08-25 | c59da8b | H20 | cuda/sm90 | GDN projections isolated | 0.00746 (512-tok, fp8) vs 0.01105 (fp4) | — | 1.48x prefill, 92.8 vs 62.6 TFLOPS |

Raw artifacts: pod `/work/before_slice4.log`, `/work/after_slice4.log`,
`/work/before2_slice4.log`, `/work/after2_slice4.log`,
`/work/dec_before_slice4.log`, `/work/dec_after_slice4.log`
(H20, GPU 3, JIT-free).
