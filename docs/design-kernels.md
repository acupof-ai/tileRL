# Kernel layer design

`packages/tilerl-kernels/src/tilerl_kernels/` is the only layer that touches TileLang or torch beyond the
tensor container. Everything above calls backend ops.

## Files

| File | Contract |
|---|---|
| `kernels.py` | Portable floor. Every kernel here compiles on CPU, and CPU is always the floor target. Holds both gemm schedules: the CPU schedule and the naive FMA fallback (used by targets whose MMA lowering rejects global operands — metal, sm90 pre-MMA). The naive kernels are a permanent fallback, not a placeholder to delete. |
| `kernels_mma.py` | SOTA copies from the tilelang ecosystem. Arch-specific lowering (sm90 today); a kernel here is allowed to not compile on CPU. Each function carries a provenance header (below). |
| `reference.py` | torch-eager reference **and** the packed-format definitions (`pack_fp4` / `unpack_fp4`). Single source of the on-disk/in-kernel format — kernels decode it, never redefine it. |
| `backend.py` | Registry only: `(precision, arch) → kernels dict`. No schedule logic. |

## Registry rules

- Arch cell = floor + overrides: `_SM90_KERNELS = {**_CPU_KERNELS, "gemm_nt": ...}`.
  Never fork the floor; override keys.
- Partial cells ship: sm90 ran on naive gemms first; MMA lands one op at a time.
- New arch = one `_register()` call. New op = floor entry + optional arch overrides.
- A kernel that can't lower on CPU goes in `kernels_mma.py`, never in the floor.

## SOTA provenance header

Every copied kernel names its source and deltas, so an upstream change can be
diffed:

```python
# SOTA copy: examples/gemm/example_gemm.py @ tilelang main
# Adapted: make_* factory signature, tileRL thread-count convention
```

## Precision selection order

Perf campaigns settle precision before tiles — skipping it wastes tile work
that a precision change invalidates:

1. **W precision** — fixed by the checkpoint (NVFP4: e2m1 packed, per-block
   scales). Not a knob.
2. **A precision** — A/B the MMA input dtype at W fixed. On sm90 the hardware
   options are e4m3 / e5m2 / bf16; Hopper wgmma needs both operands the same
   fp8 dtype, so the W dequant target follows A. Gate: relerr vs the bf16
   reference ≤ 1e-2 AND fastest.
3. **Tile/kernel** — only under the settled (W, A): block_M/N/K, K-split,
   wave count. A precision change re-opens this layer.

Decode (GEMV) gets its own pass through the same ladder — its
bandwidth/MMA balance differs from prefill.

## SOTA iteration loop

1. **FIND** — tilelang ecosystem: `examples/`, `tileop/`, `tilert/` in
   `/Users/bytedance/code/tilelang` (read-only).
2. **COPY** — into the family module (`kernels_linear.py` / `kernels_gdn.py` /
   `kernels_attn.py`; shared helpers stay in `kernels_mma.py`), adapt to the
   `make_<op>(target)` factory signature, add the provenance header.
3. **WIRE** — one line in the arch cell in `registry.py`.
4. **PARITY** — same `tests/test_ops_parity.py`, no new machinery: on CPU it
   tests the floor; on the pod it tests whatever the cell resolves. Gate:
   `allclose(rtol=1e-2)` vs torch-eager.
   `scripts/pod_sync.sh 'CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=src TILERL_TARGET=cuda python3 -m pytest tests/test_ops_parity.py -x'`
5. **BENCH** — A/B the variant against the default in one process (the ratio
   is contention-independent): a `scripts/bench_<family>.py` that builds
   inputs at canonical shapes and calls `benchkit.ab(...)`, run via
   `scripts/_pod_bench.sh` (syncs, quiet-gates GPUs 6,7, stamps the commit).
   The report prints the entry draft → `docs/experience/wins|errors/` per
   `TEMPLATE-bench.md`. No entry, not shipped.
6. **COMMIT** — `feat(ops): ...`, no AI attribution.

## Reading the emitted CUDA

Load widths, register-array shapes and `#pragma unroll` coverage are readable
off the generated source without a pod round trip.

**On the pod** — an ordinary build, then ptxas for the numbers the source
cannot show:

```bash
python3 -c 'import torch; from tilerl.ops.kernels_linear import make_linear_fp4_gemv as m
src = m("cuda", 8, 4).get_kernel_source(torch.zeros(1,17408,dtype=torch.bfloat16),
    torch.zeros(5120,8704,dtype=torch.uint8), torch.zeros(5120,1088), 32, 4, 16)
open("/work/k.cu","w").write(src)'
nvcc -arch=sm_90a -Xptxas -v -cubin /work/k.cu -o /dev/null   # registers/thread, lmem bytes
```

**On this Mac** — no GPU and no nvcc, so `scripts/cuda_codegen.py` swaps the 14
`tl.cuda.*` passes (absent from a `USE_CUDA=OFF` wheel) for identity and TVM's
stock `target.build.cuda` for tilelang's codegen entry. `enable()`, then the
same `get_kernel_source(...)`. Covers register-only kernels — the three GEMVs.
Not `T.copy` / `T.gemm`: those need `src/cuda/op/{copy,gemm}.cc`, and
`RegisterGemmImpl` takes raw C++ function pointers with no FFI, so unlike the
passes it cannot be shimmed from Python. Probe an MMA data flow through the
warp-level `mma_emitter` instead, which bypasses `tl.gemm` and does lower here.

Widths are fixed by `VectorizeLoop` before codegen, so they are safe to read;
the emitter is TVM's `CodeGenCUDA`, not tilelang's, so boilerplate is not
byte-exact. **Spills are invisible** — only ptxas or ncu sees them, and a
register array here has already fallen to local memory on the pod once
(`wins/2026-08-25-fp4-gemv-grouped-dequant.md`).

## Register-resident dequantized B

Marlin's trick — packed nibbles in shared, the dequantized weight only in
registers — is `T.gemm`'s SR variant (A shared, B fragment): `is_gemm_sr` at
`tilelang/tileop/gemm/gemm_base.py:57`, lowered by `_gemm_srr`
(`tilelang/cuda/op/gemm/gemm_mma.py:143`), which skips `ldmatrix_b` and feeds
the fragment to `mma.sync`. The B-fragment layout is inferred onto the
fragment, so a plain `T.Parallel` dequant loop lands each value in the right
lane. `example_dequant_gemm_w4a8.py:145` is the mirror image (dequantized
fragment as the first operand, RS) and allocates no shared dequant buffer; the
two examples that do are WGMMA, where shared B is a hardware requirement, and
`make_linear_fp4_mma` copied one of them.

Costs: a fragment B makes `CheckWgmma` false (`src/cuda/op/gemm.cc:40`), so
CUDA falls back to `mma.sync` — free at decode (WGMMA needs `m >= 64`, `:95`),
a real drop for prefill, which hits WGMMA today. Metal implements SS only
(`tilelang/metal/op/gemm/gemm_metal.py:97`), so the kernel needs a per-arch
cell — `_SM90_KERNELS` already is one.

Aim at the live kernel: `_CUDA_PLAN` covers all three `linear_fp4` regimes, so
`make_linear_fp4_mma`'s 8 KiB `W_shared` never runs on CUDA. The live one is
the e4m3 `W_shared` in `make_linear_fp4_fp8_mma` — 4 KiB against `WQ_shared`'s
2, `num_stages=3`: 12 KiB/CTA spent on a format conversion.

## fp4 format reconciliation

When a SOTA kernel expects a different fp4 layout/scale convention than
`pack_fp4` produces, change `pack_fp4` + its test in `reference.py` — the
format has exactly one definition. Do not teach the kernel a second format.

## Known infra debt

- Lockfile torch 2.13+cu130 vs pod driver 535: pod runs system python3.12 +
  `PYTHONPATH=src`. Fix = pin torch ≤2.11 or upgrade the driver.
- NVCC JIT per shape (30–120s): shape cache or AOT before 27B serving.
