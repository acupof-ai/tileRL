# tileRL Support Matrix

Per-op status per target. A cell that is not **done** does not work because it
compiled; `packages/tilerl-kernels/src/tilerl_kernels/registry.py` is the
source of truth and this file mirrors it.

## What "one kernel source" means

Four targets have executed the source: **cpu**, **metal**, **sm90**, **sm70**.
`_REGISTRY` holds 10 keys but 4 distinct kernel sets (each arch is registered
under both `bf16` and `fp4`, since fp4 is a weight format, not a compute dtype).
sm100/sm120 are the two empty pending-remote slots. ROCm has no cell: it was an
alias of the CPU set that never ran, removed until a HIP host runs the suite.

Cell sizes, counted off the imported module rather than the source text — a name
two cells share is only an override when the maker differs:

| cell | entries | overrides cpu | added | same maker as cpu |
| --- | ---: | ---: | ---: | ---: |
| cpu | 15 | — | — | — |
| metal | 15 | 3 (`gemm_nn/nt/tn`) | 0 | 12 |
| sm90 | 41 | 9 | 26 | 6 |
| sm70 | 23 | 2 (`silu_mul`, `gdn_prep`) | 8 | 13 |

**sm70 reuses the CPU source more than any other accelerated cell**: 13 of its
23 entries are the same maker object CPU runs, and only `silu_mul` and
`gdn_prep` are replaced. `gdn_prep` became an override because the CPU source
loops `T.serial(DK)` in every thread while the launch passes `threads=DK`, so all
128 threads computed the same 128 columns — measured at T=2048, NVH=48, DK=128,
264.33 ms against 54.04 ms for the same work at `threads=1`, and `gdn_prep` was
53.5% of a prefill tick's GPU time. sm70 now takes sm90's one-thread-per-column
schedule at f32 rather than a third copy of the kernel.
Its 8 additions are the sm70-specific decode path — `linear_fp4_gemv`,
`linear_fp4_gemv_sm70_m`, `paged_attention_split`,
`paged_attention_split_combine`, `gdn_chunk_fused`, `gdn_decode_fused`,
`rmsnorm_apply_narrow`, `write_tokens`.

Line partition of `kernels*.py` (**4,037** lines: `kernels_linear.py` 1750,
`kernels_gdn.py` 929, `kernels.py` 911, `kernels_attn.py` 290,
`kernels_mma.py` 157).

> The partition table that stood here apportioned **1,969** lines, 2.05x under the
> real count, and every share in it was derived from that figure — including the
> "1,406 lines / 71.4% sm90-only" line that has been quoted as the cost of
> supporting a second arch. It is withdrawn rather than rescaled: the shares were
> assigned by eye to whole files, and `kernels_linear.py` alone now holds both the
> sm90 WGMMA schedules and the sm70 GEMV ladder, so a per-file split cannot
> attribute it. Re-deriving it needs a per-function span walk keyed on which
> `_register` set reaches each maker.

`kernels.py` defines 25 `make_*` functions. cpu and metal reach 15 each, sm70
reaches 16, sm90 reaches 10 — sm90 is the cell that replaces the most of the
shared source, not the one that shares the most.

## Dispatch model

Kernels are registered in a `(precision, arch) -> kernel set` dict. Resolution
walks the fallback chain `exact -> (precision, "any") -> ("any", "any")`; a
registered-but-empty set raises `NotImplementedError` (pending-remote bring-up).
Adding fp8 or a new SM arch is ONE `_register()` call. Arch tags: `cpu`
(target `"c"`), `sm70`/`sm90`/`sm100`/`sm120` (CUDA capability), `metal`.

`Backend.precision` is the constant `"bf16"`
(`packages/tilerl-kernels/src/tilerl_kernels/backend.py:188`), so every row below
resolves through its arch's bf16 cell. The fp4 and fp8 tables are weight-format
tables, not separate registry cells.

**`precision` is not the tensor dtype.** It is the registry key; the dtype
kernels actually see is `Backend.io` (`backend.py:206`), which is `float32` for
cpu, metal and **sm70**, and `bfloat16` only for sm90 and later. Volta's MMA has
no bf16 path, and its kernels are written against the CPU cell's f32 parity
target, so feeding sm70 bf16 silently hands those kernels the wrong dtype. Two
cells resolve through the same `"bf16"` key and run different dtypes.

All CPU kernels are f32 compute with bf16 cast at the boundary (tilelang eager
JIT does not specialize on dtype). `linear_fp4` dequantizes on the fly.

Within a cell, `Backend` picks the linear kernel from `_CUDA_PLAN` in
`ops/backend.py` — a `(op, M-regime) -> (kernel, K pad, N cap, N tile)` table,
not a chain of string checks. The three M regimes are measured crossovers:
`M == 1` GEMV, `M <= 16` 8-way-K-split decode, above that 2-way-split prefill.
A miss falls through to the generic path; for fp8 a miss raises, because there
is no per-call fallback (see below).

## bf16

sm70 is the served arch (27B NVFP4 on a V100), so its **fwd** column is
evidenced end to end; **bwd** on sm70 has never been run and is marked
accordingly rather than inferred from the registry. A cell with no sm70 entry
resolves through the fallback chain to the CPU maker, which is how sm70 runs 14
of its 23 entries — reached, not reimplemented.

| Op | cpu | sm70 | sm90 | sm100 | metal |
| --- | --- | --- | --- | --- | --- |
| rmsnorm (fwd/bwd) | done | fwd done, bwd untested | done | pending-remote | done |
| linear (fwd/bwd) | done | fwd done, bwd untested | done | pending-remote | done |
| rope (fwd/bwd) | done | fwd done, bwd untested | done | pending-remote | done |
| attention (dense, fwd/bwd) | done | via cpu maker, untested | done | pending-remote | done |
| paged_attention (fwd) | done | done (split-KV) | done | pending-remote | done |
| gdn_forward (full GDN layer) | done | done (fused) | done | pending-remote | done |
| gdn_backward | done | untested | done | pending-remote | done |
| silu_mul (fwd/bwd) | done | fwd done, bwd untested | done | pending-remote | done |
| softmax | done | done | done | pending-remote | done |
| embedding (fwd/bwd) | done | fwd done, bwd untested | done | pending-remote | done |
| sample | done | via cpu maker | done | pending-remote | done |
| add | done | via cpu maker | done | pending-remote | done |

## fp4 (OCP e2m1 weight format)

| Op | cpu | sm70 | sm90 | sm100 | metal |
| --- | --- | --- | --- | --- | --- |
| pack_fp4 / unpack_fp4 | done | via cpu maker | done | pending-remote | done |
| linear_fp4 (fwd) | done | done | done | pending-remote | done |
| linear_fp4_gemv (M=1) | — | done | done | pending-remote | — |
| linear_fp4_gemv_sm70_m (M ladder) | — | done | — | pending-remote | — |
| linear_fp4_fp8 (w4a8, M>1) | — | — | done | pending-remote | — |

The sm70 M-ladder kernel is the one entry with no counterpart on any other arch:
`make_linear_fp4_mma8` covers the same M<=8 range on sm90 but issues
`mma.m16n8k16`, which is Ampere-and-later, so Volta reaches small-M through a
rung ladder instead
(`wins/2026-09-04-the-rung-step-is-93-percent-gemv.md`).

No cell needs a packed-weight backward kernel: training runs
`backend.linear(x, master)` on the bf16 master and its ordinary `linear_bwd`
(STE), so the tape never sees a quantized weight.

The rest of the layer (attention, norms, activations) runs the bf16 path.

## fp8 (e4m3 weight format)

| Op | cpu | sm70 | sm90 | sm100 | metal |
| --- | --- | --- | --- | --- | --- |
| linear_fp8 (native WGMMA, M>1) | dense at load | dense at load | done | pending-remote | dense at load |
| linear_fp8_gemv (M=1) | dense at load | dense at load | done | pending-remote | dense at load |
| quant_fp8 (per-token e4m3 activation) | — | — | done | pending-remote | — |

There is no `_register("fp8", ...)` cell: fp8 is a weight format inside the
bf16/fp4 cells, exactly like fp4. **dense at load** means `Backend.materialize`
rebuilds one dense weight from `.w8/.wscale/.oscale` when the cell has no fp8
kernel — once, at wiring time — in that backend's own `Backend.io` dtype. That
dtype is **not bf16 everywhere**: only sm90 has bf16 tensor cores, so
`backend.py:206` sets `io = float32` for cpu, metal **and sm70**, and bf16 only
for sm90+. Calling this row "bf16 at load" was wrong for three of the five cells.
`Backend.linear_fp8` raises
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

`save_hf` needs neither: a master-free fp4 model is written as the bytes it
serves (`.wq`/`.scale`/`.oscale` under the HF stem), so `load_hf(save_hf(m))`
is bit-identical at whatever block size the model held — no dequant on save,
no re-pack on load. Where a master IS present it is the live weight the
optimizer moves, so it is saved alone and the stale bytes are dropped. Keys
with neither (fused projections, native-fp8 serving) still raise.

## Evidence

- **cpu/bf16 + fp4 + fp8**: `TILERL_TARGET=cpu uv run pytest` — 97 passed,
  4 skipped: kernel-vs-reference parity on every op with a CPU kernel, tape
  gradcheck, end-to-end generation + training. Dense/paged attention's CPU
  forward is the torch-eager reference itself (`backend.attention`, ponytail),
  so attention parity runs only on sm90.
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
  — one fp4 tiny model, 8 greedy tokens on cpu and on metal through the
  engine: identical token ids, `allclose(rtol=1e-2)` prefill logits.
  Auto-skips where MPS is unusable (CI runners).
- **metal/bf16 + fp4**: `TILERL_TARGET=metal uv run pytest` — 97 passed, 4
  skipped. Same kernel source as CPU; the only registry fork is the three
  gemms (naive FMA — Metal's `T.gemm` lowering rejects global operands).
  `tilerl bench` tiny: prefill 191 tok/s, decode 116 on Metal vs 1202 / 705
  on CPU — every op migrates CPU-resident params to MPS, and the tiny model is
  dispatch-bound. `# ponytail: per-op param migration, keep params on the backend device`
  tilelang 0.1.13 Metal kernel-cache save is broken (`MetalKernelAdapter` has
  no `libpath`) — kernels recompile per process.
- **sm90/bf16 + fp4 + fp8**: `TILERL_TARGET=cuda uv run pytest` on an H20 —
  60 passed, 3 metal-only skips (2026-08-24, not re-run since).
  `Backend.device` pins `cuda:<current>` (an index-less `torch.device("cuda")`
  is not where kernel outputs land). The 2-layer Qwen3.6-27B NVFP4 slice
  forwards and trains with CPU/CUDA logits matching to 6 decimals
  (`docs/experience/wins/2026-08-24-sm90-real-slice.md`).
- **sm70/bf16 + fp4**: the served path, not a test-suite row. `TILERL_TARGET=cuda
  uv run pytest` has never been run on the V100; what is evidenced is the 27B
  NVFP4 checkpoint serving end to end — `docs/serve-v100.md`, measured 2026-09-05
  at **50.3 tok/s decode-only / 46.3 wall, ttft 324 ms**, ctx 4096, B=1, 22 GiB.
  So `fwd` above means "this op runs inside a served token" and `bwd` means
  nothing has exercised it. `Backend.io` is **f32** here, not bf16 (see Dispatch
  model); building the kernels needs cuda-12.4's nvcc.
- **sm100/sm120**: registered as empty sets, `NotImplementedError` on use;
  bring-up pending-remote.
