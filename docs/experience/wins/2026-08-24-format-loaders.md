# Per-format loader tests, 23MB fixture retired — cpu, 2026-08-24

> Status: Shipped

## Context

`load_hf` had four checkpoint formats to support for the 27B family: bf16 HF
(existing), ModelOpt NVFP4/FP8-block (existing), official NVIDIA NVFP4 naming
(the deployment target `nvidia/Qwen3.6-27B-NVFP4`), per-tensor FP8, and
AWQ-int4. The only real-weight coverage was a committed 23MB MLX fixture
(`tests/fixtures/qwen35-2layer-mlx4/`), cropped from a local 0.8B model — the
suite's sole non-synthetic dependency, and it covered only the MLX path.

## What Worked

Format dispatch by sibling-tensor presence, one branch per format, all
dequantizing to bf16 before param mapping (the model stays format-agnostic):

- `weight_packed` → ModelOpt NVFP4 (pre-existing).
- `weight_scale_2` sibling → official NVFP4 MLP — same e2m1×e4m3×global math
  as ModelOpt, reuses `dequant_nvfp4`; only the tensor names differ.
- scalar `weight_scale` sibling → per-tensor FP8 (official NVFP4 GDN/attn
  linears, standalone FP8) — one `w.float() * scale` line.
- `qweight` sibling → AWQ-int4 via new `dequant_awq` (autoawq GEMM packing:
  8 int4 per int32 for 8 consecutive output features, [K,N//8] → [N,K]).

Each format got one KB-sized synthetic test in `tests/test_weights.py`
following the ModelOpt pattern: random packed bytes, a formula reference
written independently from the dequant implementation, `torch.equal` on the
loaded bf16. The 23MB fixture, `test_real_weights.py`, and `crop_fixture.py`
were deleted; the MLX path kept coverage via `test_mlx_affine_load`. Suite:
62 → 64 hermetic tests, all green on CPU.

## Rule

A checkpoint format is "supported" when it has a sibling-detection branch, a
dequant to bf16, and a synthetic test asserting against an independent formula
reference — never a large binary fixture. Real-weight validation is a
manual/pending-remote step, not a suite dependency.
