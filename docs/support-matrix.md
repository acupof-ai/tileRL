# tileRL Support Matrix

Canonical support-status truth for the tileRL kernel layer. If a cell is not
**done**, do not assume it works because it compiled — the dispatch registry
(`src/tilerl/ops/backend.py`) is the source of truth, and this file mirrors it.
State as of 2026-08-24.

## Dispatch model

Kernels are registered in a `(precision, arch) -> kernel set` dict. Resolution
walks the fallback chain `exact -> (precision, "any") -> ("any", "any")`; a
registered-but-empty set raises `NotImplementedError` (pending-remote bring-up).
Adding fp8 or a new SM arch is ONE `_register()` call. Arch tags: `cpu`
(target `"c"`), `sm90`/`sm100`/`sm120` (CUDA capability), `rocm`, `metal`.

All CPU kernels are f32 compute with bf16 cast at the boundary (tilelang eager
JIT does not specialize on dtype). fp4 is a weight format, not a compute dtype:
its cell reuses the bf16 kernel set (`linear_fp4` dequantizes on the fly).

## bf16

| Op | cpu | sm90 | sm100 | rocm | metal |
| --- | --- | --- | --- | --- | --- |
| rmsnorm (fwd/bwd) | done | pending-remote | pending-remote | pending-remote | done |
| linear (fwd/bwd) | done | pending-remote | pending-remote | pending-remote | done |
| rope (fwd/bwd) | done | pending-remote | pending-remote | pending-remote | done |
| attention (dense, fwd/bwd) | done | pending-remote | pending-remote | pending-remote | done |
| paged_attention (fwd) | done | pending-remote | pending-remote | pending-remote | done |
| linear_attn_chunk (plain scan) | done | pending-remote | pending-remote | pending-remote | done |
| gdn_forward (full GDN layer) | done | pending-remote | pending-remote | pending-remote | done |
| gdn_backward | done | pending-remote | pending-remote | pending-remote | done |
| silu_mul (fwd/bwd) | done | pending-remote | pending-remote | pending-remote | done |
| softmax | done | pending-remote | pending-remote | pending-remote | done |
| embedding (fwd/bwd) | done | pending-remote | pending-remote | pending-remote | done |
| sample | done | pending-remote | pending-remote | pending-remote | done |
| add | done | pending-remote | pending-remote | pending-remote | done |

## fp4 (e2m1 weight format)

| Op | cpu | sm90 | sm100 | rocm | metal |
| --- | --- | --- | --- | --- | --- |
| pack_fp4 / unpack_fp4 | done | pending-remote | pending-remote | pending-remote | done |
| linear_fp4 (fwd) | done | pending-remote | pending-remote | pending-remote | done |
| linear_fp4_bwd (STE) | done | pending-remote | pending-remote | pending-remote | done |

The rest of the layer (attention, norms, activations) runs the bf16 path.

## fp8

Not started — no `_register("fp8", ...)` call exists yet. The 27B target is
NVFP4, so fp8 is not on the day-1 path; add the registry entry + kernels when
a checkpoint demands it.

## Evidence

- **cpu/bf16**: `uv run pytest` (58 hermetic tests) — kernel-vs-reference
  parity on every op, tape gradcheck, end-to-end generation + training.
- **cpu/fp4**: `test_fp4_on_load_and_forward` (pack-on-load + forward through
  the fp4 linear), `test_fp4_roundtrip` (wire format), STE backward covered by
  the production-model gradcheck.
- **real weights**: `TILERL_TEST_REAL=1 uv run pytest tests/test_real_weights.py`
  — Qwen3.5-0.8B-MLX-4bit forward + train_step; full 24-layer generation is
  coherent ("The capital of France is Paris, and the capital of the United
  States is Washington, D.C.").
- **metal/bf16 + fp4**: `TILERL_TARGET=metal uv run pytest` — the same 58 tests
  pass on Apple Silicon (tilelang 0.1.13 Metal JIT, torch MPS), plus
  `tests/test_metal_target.py` (target resolution, rmsnorm, the metal gemm
  schedule). `TILERL_TARGET=metal TILERL_TEST_REAL=1 uv run pytest
  tests/test_real_weights.py` passes (RefBackend path — validates device
  wiring, not metal kernels). Metal facts:
  - Same kernel source as CPU; the only registry fork is the three gemms
    (naive FMA schedules — Metal's `T.gemm` lowering rejects global operands).
  - `linear_attn_chunk` uses a per-column serial scan on BOTH targets: the
    shared-memory + `T.Parallel`-column schedule is nondeterministic on Metal
    (tilelang 0.1.13 codegen races cross-loop shared visibility; observed
    drift 2.4e-2 across identical runs). The serial scan is drift-free.
  - `tilerl bench` (tiny, steady state after shader-cache warmup): prefill
    191 tok/s, decode 116 tok/s on Metal vs 1202 / 705 tok/s on CPU — Metal is
    ~6x slower at this scale because every op migrates CPU-resident params to
    MPS (day-1 boundary design). Metal's target is the 27B, which cannot run
    on CPU; the tiny model is latency-bound on dispatch, not compute.
    `# ponytail: per-op param migration, keep params on the backend device`
  - tilelang 0.1.13 Metal kernel-cache save is broken (`MetalKernelAdapter`
    has no `libpath`; non-fatal ERROR log) — kernels recompile per process.
- **GPU arches (sm90/sm100/sm120/rocm)**: registry slots registered as empty
  sets; `NotImplementedError` on use. Bring-up is pending-remote (no CUDA/ROCm
  device in this env).
