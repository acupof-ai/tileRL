# bf16 GEMV decode kernel: 42-116% roof on bf16 projections — but the 27B decode is all-fp4 (premise correction) — H20, 2026-08-25

> Status: Shipped

## Context

The SOTA-all-levers bench (2026-08-24-sota-all-levers.md) named the bf16 GEMV
"the biggest single lever" for the 27B decode: it claimed the GDN projections
are bf16 weights (73% of decode bytes) running M=1 padded WGMMA at 10-15% of
HBM roofline, and that a bf16 GEMV (the fp4 GEMV schedule minus the dequant)
would close most of the 4.7x gap to the 80 tok/s target.

This entry ships that kernel (`make_linear_bf16_gemv`) and corrects the
premise: on the fp4 27B (`cfg.fp4=True`), `load_hf` packs EVERY projection —
GDN `in_proj_*`/`out_proj`, MLP, `lm_head` — to fp4 (`fp4_param_keys` has
included the GDN projections since the loader's creation). The decode
per-op profile shows only `linear_fp4` (86% of GPU sum), zero bf16 `linear`
ops. The bf16 masters in `model.params` are the STE/training copy; the engine
reads the `.wq`. So the bf16 GEMV does not move the 27B decode — it serves
non-fp4 models (`cfg.fp4=False`) and is the bf16 counterpart to the fp4 GEMV.

## What Worked

**Kernel.** `make_linear_bf16_gemv` mirrors `make_linear_fp4_gemv` (split-K +
warp allreduce, one warp group per 4 output rows, `micro_size_k=8` bf16 = the
128-bit transaction) with the dequant stage removed: bf16 W streamed
directly, f32 accumulate of the bf16 products (exactly what WGMMA does —
bf16*bf16 is exact in f32), f32 output to match the WGMMA path. Roofline =
(N*K*2 + 2K) bytes / HBM BW. Parity: local 26 passed, pod CUDA 28 passed
(allclose rtol=1e-2 vs `reference.linear`, tiny shapes).

**Per-linear roofline (slice2 bf16 masters, BW 3308.7 GB/s, contended pod):**

| shape (N,K) | projection | bytes MB | GEMV ms | WGMMA ms | GEMV %roof | WGMMA %roof | speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| 10240,5120 | in_proj_qkv | 100.0 | 0.0456 | 0.1172 | 69.5% | 27.1% | 2.6x |
| 6144,5120 | in_proj_z | 60.0 | 0.0446 | 0.0856 | 42.7% | 22.2% | 1.9x |
| 5120,6144 | out_proj | 60.0 | 0.0458 | 0.0953 | 41.6% | 19.9% | 2.1x |
| 17408,5120 | gate/up_proj | 170.0 | 0.0529 | 0.1540 | 101.9% | 35.0% | 2.9x |
| 5120,17408 | down_proj | 170.0 | 0.0638 | 0.2474 | 84.5% | 21.8% | 3.9x |
| 248320,5120 | lm_head | 2425.0 | 0.6615 | 1.8264 | 116.2% | 42.1% | 2.8x |
| 48,5120 | in_proj_b/a | 0.5 | 0.0569 | 0.0854 | 0.3% | 0.2% | 1.5x |

The bf16 GEMV is at 42-116% of roof on every projection that matters (the
N=48 rows are 0.5 MB — launch-latency-bound for both kernels, negligible
bytes). The padded-M=16 WGMMA path it replaces is at 20-42%. The bf16 GEMV
is also 2-3x MORE roof-efficient than the fp4 GEMV (24-33%): same schedule,
the only difference is the dequant stage — the fp4 nibble-decode + per-tile
scale is what caps the fp4 kernel, not the GEMV schedule.

**Slice decode before/after (slice2, 30-tick avg, graph-captured wall):**
BEFORE (WGMMA) 1.932 ms/tick (517.6 tok/s) → AFTER (GEMV) 1.922 ms/tick
(520.2 tok/s). No change — the expected negative result. The decode path
calls `linear_fp4` for every projection (the `.wq` exists), so
`backend.linear` — and thus the bf16 GEMV — is never exercised. The "linear"
op count is 0 in both phases.

## Rule

Verify which weights are actually bf16 vs fp4 in the decode path before
building the bf16 path: `model.params` carries BOTH the bf16 master and the
fp4 `.wq`/`.scale`, and the engine reads the `.wq` (`model._linear`
dispatches on `key + ".wq" in params`). The SOTA-all-levers roofline (30.9
GB, 73% bf16) counted the bf16 masters; the engine actually streams the fp4
`.wq` (~19 GB/tick by the same model). And: the bf16 GEMV at 42-116% roof
proves the split-K + warp-allreduce schedule is sound and BW-saturating —
the fp4 GEMV's 24-33% is the dequant stage, the real 27B decode lever.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | eb4a463 | H20 | cuda/sm90 | 27B slice (2 GDN) | — | 1.922 (after) / 1.932 (before) | 520 / 518 decode |

Decode ms/tick is the graph-captured per-tick wall (30-tick avg); before =
the GEMV key popped from the sm90 cell (M=1 pads to 16 WGMMA rows), after =
GEMV dispatch. The 0.5% delta is noise — the fp4 27B decode path does not
call `backend.linear`. Per-linear roofline: GEMV 42-116% vs WGMMA 20-42%
(1.9-3.9x) on the bf16 masters.

Raw artifacts: pod `/work/bench_bf16_gemv_slice2.log`,
`/work/parity_bf16_gemv.log` (H20, GPU 1, JIT-free).
