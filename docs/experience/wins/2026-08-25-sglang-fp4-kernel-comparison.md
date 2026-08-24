# sglang NVFP4 kernel source vs tileRL fp4 GEMV — what they do, what to copy — H20 pod, 2026-08-25

> Status: Shipped

## Context

The final 80/3800 bench (2026-08-25-bf16-gemv-fp8-weights.md) verdict: the
2.25x decode / 3.8x prefill gap to target is fp4 dequant efficiency, not
physics — the fp4 GEMV sits at 24-33% of HBM roof while the bf16 GEMV on the
same schedule hits 42-116%. Before optimizing the dequant, read how the
framework we benchmark against implements NVFP4: its checkpoint format, its
GEMM dispatch by arch, where the dequant lives, and what (if anything) is
copyable into tileLang. This is a source read only — no measurement.

## Sources & versions

- Installed sglang: `0.5.13.post2.dev55+g0e7e68b76` (bytedance-iaas fork,
  editable at `/sgl-workspace/sglang`). Its `srt/layers/quantization/` is
  stripped to awq / compressed_tensors (empty schemes) / modelslim / quark —
  no NVFP4 scheme ships in the installed fork at all.
- Upstream reference tree: `/host/sglang-sync2/python/sglang/` (full upstream
  main snapshot, the source of every sglang path cited below).
- flashinfer-python `0.6.11.post1+cu129` at
  `/usr/local/lib/python3.12/dist-packages/flashinfer/` (the GEMM backend
  sglang dispatches to).
- tileRL: `src/tilerl/ops/kernels_mma.py` (`make_linear_fp4_gemv` :215,
  `make_linear_fp4_fp8_mma` :482), `src/tilerl/ops/reference.py`
  (`pack_fp4` :227, `dequant_nvfp4` :271).

## How sglang does NVFP4

**Format** (`srt/layers/quantization/compressed_tensors/schemes/
compressed_tensors_w4a4_nvfp4.py`). W4A4 throughout:

- `weight_packed` uint8 `[N, K//2]` — two OCP/MX e2m1 nibbles per byte (the
  grid *with* zero, `{0,.5,1,1.5,2,3,4,6}`), low nibble first.
- `weight_scale` **e4m3** `[N, K//16]` — per-16-block scale factor stored as
  FP8, not f32.
- `weight_global_scale` f32 per tensor; `input_global_scale` f32 per tensor.
  Activations are quantized to FP4 too. `alpha = 1/(gs_a * gs_w)` is the only
  f32 the GEMM epilogue applies.
- `get_min_capability() = 100` — this scheme **refuses to load on Hopper**.
  The modelopt NVFP4 path (`modelopt_quant.py:419`, `ModelOptFp4Config`)
  sets `89` and is the one that can run on H20.

**GEMM dispatch** (`modelopt_quant.py:127` `fp4_gemm` → flashinfer
`mm_fp4`, `flashinfer/gemm/gemm_base.py:5794`). Backend by arch:

| arch | auto picks | kernel |
|---|---|---|
| sm120 | cudnn | cuDNN (closed) |
| sm100/103 | cute-dsl / cutlass | CUTLASS native FP4 tensor core |
| sm90 (H20) | cudnn | cuDNN (closed) — the cutlass backend raises `ValueError` on sm90 (`gemm_base.py:1384`, sm100/103/120 only); trtllm is trtllm-gen (Blackwell); sglang's own JIT cutlass kernel (`jit_kernel/csrc/gemm/nvfp4/`) ships only `_sm100`/`_sm120` specializations |

So on Hopper, sglang's NVFP4 path is cuDNN's closed GEMM. There is **no
open-source Hopper FP4 GEMM in the sglang/flashinfer tree**. The open Hopper
reference for the dequant-in-mainloop pattern is TRT-LLM's W4A16 mixed-input
dqMMA (`flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/
fpA_intB_gemm/fpA_intB_gemm_template_sm90.h`): int4/fp4 weights, bf16
activations, dequant fused into the CUTLASS mainloop with TMA + warp
specialization.

**Blackwell kernel** (`nvfp4_scaled_mm_sm100.cuh`). CUTLASS
`OpClassBlockScaledTensorOp` with `nv_float4_t<e2m1>` operands and ue4m3
block scales — **no software dequant at all**, the tensor core consumes fp4
+ scale factors directly (block-scaled FP4 MMA). TMA warp-specialized
1Sm/2Sm, `MmaTileShape` 128×256×256, cluster (1,4,1)/(2,4,1), tile configs
bucketed by M (M≤128 → 1Sm; M≤256 → 2Sm; M≤1024 → 2Sm 2×4; M>1024 → 1×4).
Scale layout is the 128×4 swizzle (`utils.py:595` `swizzle_blockscale`:
pad to [N↑128, K↑4], reshape (rows//128, 4, 32, cols//4, 4), permute)
matching the CUTLASS SF tile atom.

**Activation quant** (`nvfp4_quant_kernels.cuh` `cvt_warp_fp16_to_fp4`).
Per-warp 16-elem block (2 threads × 8 half2), warp-shuffle max reduction,
`SF = gs * vecmax/6` stored as e4m3, reciprocal output scale applied,
`fp32_vec_to_e2m1`. Guarded `__CUDA_ARCH__ >= 1000` — sglang's own quant is
Blackwell-only; on Hopper flashinfer's cuda `fp4_quantize` is used.

**No dedicated decode GEMV.** M=1 goes through the same `fp4_gemm` with the
small-M tile config. sglang does not special-case decode for NVFP4.

## How tileRL does it now

- **Format** (`reference.py:227` `pack_fp4`): e2m1**fn** grid (no zero),
  per-**32**-block **f32** scale (`block_max/6`, LUT argmin round-to-nearest),
  no global scale. The checkpoint loader (`dequant_nvfp4` :271) converts the
  checkpoint's OCP e2m1 + e4m3-per-16 + global-scale wire format to this
  internal format at load time.
- **Decode** (`kernels_mma.py:215` `make_linear_fp4_gemv`): W4A16 split-K
  GEMV — one warp group per 4 output rows, `micro_size_k=8` (128-bit
  transaction / 16-bit bf16), WQ streamed as uint8, one f32 scale lookup per
  micro-tile, warp allreduce. Dequant is per-element serial in the FMA loop:
  nibble extract → integer bit-pattern fast decode
  (`sign<<31 | (126+e)<<23 | m<<22`, reinterpret as fp32 — the lop3-style
  trick, no LUT load, no exp2) → `×scale` → FMA. 24-33% of roof (3310 GB/s
  measured); the bf16 GEMV on the identical schedule hits 42-116%.
- **Prefill** (`kernels_mma.py:482` `make_linear_fp4_fp8_mma`): W4A8 —
  dequant e2m1fn→fp32→×scale→requant e4m3 in the K-loop, fp8 WGMMA. 21% of
  peak; the dequant-to-e4m3 cast is the cap (62% of the prefill tick).

## Side by side

| dimension | sglang NVFP4 | tileRL |
|---|---|---|
| weight grid | OCP/MX e2m1 (with zero) | e2m1fn (no zero) |
| block size | 16 | 32 |
| block scale dtype | e4m3 (1 B) | f32 (4 B) |
| global scale | yes, per-tensor f32 (`alpha`) | no (absorbed per block) |
| scale layout | 128×4 swizzled (CUTLASS SF atom) | plain `[N, K//32]` |
| activations | W4A4 (fp4, per-16-block e4m3 SF) | W4A16 decode / W4A8 prefill |
| Hopper MMA | cuDNN GEMM, dequant→bf16 in mainloop (closed) | bf16 FMA in GEMV (no tensor cores) |
| Blackwell MMA | native block-scaled FP4 tensor core, zero software dequant | n/a (sm90 target) |
| dequant location | fused into pipelined GEMM mainloop (TMA, warp-spec) | per-element, serial with the FMA |
| decode M=1 | same GEMM, small-M tile config | dedicated split-K GEMV |

## What to copy (ranked by expected impact)

1. **Move the dequant off the FMA critical path.** This is the whole gap.
   Same schedule, bf16 GEMV 42-116% roof vs fp4 GEMV 24-33%: the per-element
   dequant (~6 int ops + reinterpret + fmul) is serialized with the
   loop-carried FMA dependency. The dqMMA reference (TRT-LLM
   `fpA_intB_gemm_template_sm90.h`) fuses dequant into a pipelined mainloop
   stage — load, dequant ahead of the consume, MMA/FMA on already-dequanted
   data. In the tileRL GEMV this means: software-pipeline the K-loop
   (dequant micro-tile ko+1 while ko's FMA runs, double-buffered in
   registers), or at minimum split the loop body into dequant-all-8 then
   FMA-all-8 so the dequant ILP is not on the FMA dep chain.
2. **Vectorize the nibble unpack 8-wide.** Today each element re-loads its
   shared byte (`WQ_local[ki//2]`) and decodes scalar. A uint32 load (8
   nibbles) + parallel mask/shift extracts all 8 at once; the integer
   bit-pattern construction — already the right trick vs LUT/exp2 — then
   issues 8-wide instead of 1-wide.
3. **e4m3 block scales.** f32 per-32 = 0.125 B/elem, ~20% of the
   WQ+scale stream; e4m3 = 0.031 B/elem, ~15% less traffic on a
   memory-bound kernel. Needs a per-tensor global scale to recover range
   (sglang's `alpha`); tileRL's `block_max/6` already normalizes per block,
   so the SF stores `block_max/(6·gs)`. Bonus: the NVFP4 checkpoint already
   ships e4m3 per-16 SF — keeping them e4m3 end-to-end deletes the
   `dequant_nvfp4` f32 conversion too.

## What not to copy

- **GEMM-for-decode.** sglang runs M=1 through the GEMM (128×256 tile).
  tileRL's dedicated split-K GEMV is structurally better for M=1; the gap is
  dequant efficiency, not the schedule. Keep the GEMV.
- **W4A4 / native FP4 MMA.** Blackwell-only. On sm90, W4A4 buys nothing for
  decode (activations are ~2 KB/tick, negligible) and the prefill W4A8 path
  already exists. Revisit only if a Blackwell target appears — the e4m3 SF +
  128×4 swizzle is the format to keep compatible.

## Rule

sglang's NVFP4 on Hopper is a closed cuDNN GEMM doing the same
dequant→bf16→MMA tileRL does — its edge is a pipelined mainloop with the
dequant off the MMA critical path, not a different decode math. Copy the
pipelining and the 8-wide nibble unpack, keep the GEMV schedule, and take
e4m3 block scales as the cheap third lever.
