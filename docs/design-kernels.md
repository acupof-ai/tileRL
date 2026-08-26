# Kernel layer design

`src/tilerl/ops/` is the only layer that touches TileLang or torch beyond the
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

## fp4 format reconciliation

When a SOTA kernel expects a different fp4 layout/scale convention than
`pack_fp4` produces, change `pack_fp4` + its test in `reference.py` — the
format has exactly one definition. Do not teach the kernel a second format.

## Known infra debt

- Lockfile torch 2.13+cu130 vs pod driver 535: pod runs system python3.12 +
  `PYTHONPATH=src`. Fix = pin torch ≤2.11 or upgrade the driver.
- NVCC JIT per shape (30–120s): shape cache or AOT before 27B serving.
