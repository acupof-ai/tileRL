# tileRL Support Matrix

Canonical support-status truth for the tileRL kernel layer. If a cell is not
**done**, do not assume it works because it compiled — the dispatch registry
(`src/tilerl/ops/registry.py`) is the source of truth, and this file mirrors it.
State as of 2026-08-26.

## Dispatch model

Kernels are registered in a `(precision, arch) -> kernel set` dict. Resolution
walks the fallback chain `exact -> (precision, "any") -> ("any", "any")`; a
registered-but-empty set raises `NotImplementedError` (pending-remote bring-up).
Adding fp8 or a new SM arch is ONE `_register()` call. Arch tags: `cpu`
(target `"c"`), `sm90`/`sm100`/`sm120` (CUDA capability), `rocm`, `metal`.

`rocm` shares the CPU cell rather than sitting in an empty slot: the schedules
are block-parallel and target-neutral, so the same source compiles for HIP. No
HIP host exists in this env, so the cells below read **untested**, not done.

All CPU kernels are f32 compute with bf16 cast at the boundary (tilelang eager
JIT does not specialize on dtype). fp4 is a weight format, not a compute dtype:
its cell reuses the bf16 kernel set (`linear_fp4` dequantizes on the fly).

Within a cell, `Backend` picks the linear kernel from `_CUDA_PLAN` in
`ops/backend.py` — a `(op, M-regime) -> (kernel, K pad, N cap, N tile)` table,
not a chain of string checks. The three M regimes are measured crossovers:
`M == 1` GEMV, `M <= 16` 8-way-K-split decode, above that 2-way-split prefill.
A miss falls through to the generic path; for fp8 a miss raises, because there
is no per-call fallback (see below).

## bf16

| Op | cpu | sm90 | sm100 | rocm | metal |
| --- | --- | --- | --- | --- | --- |
| rmsnorm (fwd/bwd) | done | done | pending-remote | untested (CPU cell) | done |
| linear (fwd/bwd) | done | done | pending-remote | untested (CPU cell) | done¹ |
| rope (fwd/bwd) | done | done | pending-remote | untested (CPU cell) | done |
| attention (dense, fwd/bwd) | done | done | pending-remote | untested (CPU cell) | done |
| paged_attention (fwd) | done | done | pending-remote | untested (CPU cell) | done |
| linear_attn_chunk (plain scan) | done | done | pending-remote | untested (CPU cell) | done |
| gdn_forward (full GDN layer) | done | done | pending-remote | untested (CPU cell) | done |
| gdn_backward | done | done | pending-remote | untested (CPU cell) | done |
| silu_mul (fwd/bwd) | done | done | pending-remote | untested (CPU cell) | done |
| softmax | done | done | pending-remote | untested (CPU cell) | done |
| embedding (fwd/bwd) | done | done | pending-remote | untested (CPU cell) | done |
| sample | done | done | pending-remote | untested (CPU cell) | done |
| add | done | done | pending-remote | untested (CPU cell) | done |

¹ `linear(x, w, bias)` fails on metal: the bias is not migrated to the backend
device, so `gemm_nt` rejects it (`device_type mismatch`). Pre-existing and
boundary-only — the biasless path, which the model uses everywhere, is green.

## fp4 (OCP e2m1 weight format)

| Op | cpu | sm90 | sm100 | rocm | metal |
| --- | --- | --- | --- | --- | --- |
| pack_fp4 / unpack_fp4 | done | done | pending-remote | untested (CPU cell) | done |
| linear_fp4 (fwd) | done | done | pending-remote | untested (CPU cell) | done |
| linear_fp4_gemv (M=1) | — | done | pending-remote | — | — |
| linear_fp4_fp8 (w4a8, M>1) | — | done | pending-remote | — | — |

No cell needs a packed-weight backward kernel: training runs
`backend.linear(x, master)` on the bf16 master and its ordinary `linear_bwd`
(STE), so the tape never sees a quantized weight.

The rest of the layer (attention, norms, activations) runs the bf16 path.

## fp8 (e4m3 weight format)

| Op | cpu | sm90 | sm100 | rocm | metal |
| --- | --- | --- | --- | --- | --- |
| linear_fp8 (native WGMMA, M>1) | bf16 at load | done | pending-remote | bf16 at load | bf16 at load |
| linear_fp8_gemv (M=1) | bf16 at load | done | pending-remote | bf16 at load | bf16 at load |
| quant_fp8 (per-token e4m3 activation) | — | done | pending-remote | — | — |

There is no `_register("fp8", ...)` cell: fp8 is a weight format inside the
bf16/fp4 cells, exactly like fp4. **bf16 at load** means `Backend.materialize`
rebuilds one bf16 weight from `.w8/.wscale/.oscale` when the cell has no fp8
kernel — once, at wiring time. `Backend.linear_fp8` raises
`NotImplementedError` naming the cell if it is ever reached without that
conversion; it never falls back to a master weight per call.

## fp4 weight representation

`load_hf` serves the checkpoint's bytes; there is no bf16 round-trip.

| Key | Type | Meaning |
| --- | --- | --- |
| `<key>.wq` | uint8 `[N, K/2]` | OCP e2m1 nibbles, low nibble first — the checkpoint's bytes verbatim |
| `<key>.scale` | f32 `[N, K/B]` | block scale; **B comes from the checkpoint** (16 for NVFP4, 32 when `pack_fp4` produced it) |
| `<key>.oscale` | f32 `[N]`, optional | per-output-row epilogue scale |

`y = oscale[n] * sum_k x[k] * e2m1(wq) * scale[n, k/B]`. B is a call-time
kernel parameter, not a registry entry — the backend derives it as
`K // scale.shape[1]`.

OCP e2m1 (`[0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6]`) is the one internal grid, so
the checkpoint grid and the kernel grid are the same thing and padded nibbles
(`0x00`) decode to 0.0.

`.scale` stays f32 rather than native e4m3 bytes on purpose: the 2026-08-25
e4m3-scale A/B was a 5-11% decode regression
(`docs/experience/wins/2026-08-25-fp4-e4m3-block-scales.md`). Native-e4m3
storage is a pending-remote A/B, not part of the current design.

`.scale` is renormalized at load so every row's `6 * scale` lands in `[6, 12)`
(the removed factor is an exact power of two, carried in `.oscale`). The w4a8
kernel dequantizes into e4m3, whose usable range is 448 with subnormals below
2^-9; unnormalized checkpoint magnitudes saturate (50% weight error) and
`block_max/6` weight units collapse into subnormals (3.8%). Renormalized, the
error is 2.3% — e4m3's own 3-mantissa-bit requant floor.

## The bf16 master is training-only

`load_hf(..., keep_master=True)` regenerates `params[key]` from the served
quantized bytes for the tape's STE gradient. Serving (`tilerl serve`,
`tilerl bench`) takes the default `False` and ships no master to the device.
For the 27B that is ~51 GB of bf16 that never leaves the loader.

`save_hf` needs masters and raises without them, rather than writing a shard
with no linear weights.

## Evidence

- **cpu/bf16 + fp4 + fp8**: `TILERL_TARGET=cpu uv run pytest` — 96 passed, 4
  skipped. Kernel-vs-reference parity on every op with a TileLang CPU kernel,
  tape gradcheck, end-to-end generation + training. Exception: dense/paged
  attention's CPU forward is the torch-eager reference itself
  (`backend.attention`, ponytail — no TileLang CPU attention kernel yet), so
  attention has no independent CPU parity until that kernel lands; sm90
  attention parity runs on the pod.
- **fp4 grid**: `test_linear_fp4_grid` feeds the *kernel* a literal OCP table
  through `x = eye(K)`, so it shares no constant with `pack_fp4` /
  `dequant_fp4` — the only fp4 test in the suite that can catch a wrong grid.
  Runs at B=32 and B=16.
- **w4a8 range**: `test_fp4_w4a8_e4m3_range` casts the dequantized weight to
  e4m3 the way `kernels_linear.py` does and asserts <=3% relative error on a
  simulated NVFP4 tensor. The one CPU-runnable gate on a GPU-only numerics
  failure.
- **checkpoint loaders**: `tests/test_weights.py` — one synthetic hermetic
  test per format: bf16 HF roundtrip, ModelOpt NVFP4 + FP8-block, official
  NVFP4 (MLP e2m1 + GDN/attn per-tensor and per-channel FP8), AWQ-int4, MLX
  affine-4bit. The NVFP4 test runs both `fp4=False` (dequant correctness) and
  `fp4=True` (`.wq` byte-identical to the checkpoint).
- **heterogeneity**: `tests/test_metal_target.py::test_cpu_metal_decode_parity`
  — one fp4 tiny model, greedy-decoded 8 tokens on cpu and on metal through
  the engine: identical token ids and `allclose(rtol=1e-2)` prefill logits.
  This is the gate T6 asks for; it auto-skips where MPS is unusable (ubuntu
  CI, macos CI runners with no GPU entitlement).
- **rocm**: `tests/test_ops_parity.py::test_rocm_cell_is_the_cpu_cell` — a
  dict lookup, so the matrix claim is gated on every host. Nothing has ever
  run on HIP.
- **metal/bf16 + fp4**: `TILERL_TARGET=metal uv run pytest` — 94 passed, 4
  skipped, 2 failed: `test_linear_parity` (the bias-migration bug above) and
  `test_weights.py::test_fused_projections_parity` (the same class — the test
  mixes CPU-resident and mps tensors in a reference call). Both are boundary
  bugs in test/host code, not target-neutrality bugs, and both reproduce on
  the pre-refactor tree. `tests/test_metal_target.py` itself is green (4
  passed). Metal facts:
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
- **sm90/bf16 + fp4 + fp8**: `TILERL_TARGET=cuda uv run pytest` on an H20
  (pod, tilelang 0.1.13 CUDA JIT) — the suite passes green (60 passed, 3
  metal-only skips, as of 2026-08-24; not re-run since). `Backend.device`
  pins `cuda:<current>` (`torch.device("cuda")` with no index is not the
  device kernel outputs land on). First real-weight run: the 2-layer
  Qwen3.6-27B NVFP4 slice forwards and trains on sm90, with CPU/CUDA
  logits matching to 6 decimals (entry:
  `docs/experience/wins/2026-08-24-sm90-real-slice.md`).
- **sm100/sm120**: registry slots registered as empty sets;
  `NotImplementedError` on use. Bring-up is pending-remote (no device in
  this env).
