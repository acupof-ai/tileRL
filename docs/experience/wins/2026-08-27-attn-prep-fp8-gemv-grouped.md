# attn_prep fusion + grouped fp8 GEMV — cuda(H20), 2026-08-27

> Status: Shipped

## Context

Decode B=1 tick 19.3 ms (51.9 tok/s) on the now-correct 27B. Two changes,
A/B'd on a quiet host with `scripts/bench_harness.py --suite decode-kv`
(median of 3×20 ticks, engine rebuilt per row).

## What Worked

1. **`attn_prep`** (`kernels_mma.py`): per full-attn layer, q_norm + k_norm +
   partial RoPE + paged K/V write in one launch straight off the fused-qkv
   GEMV output (no q/k slice copies, no rope cat). Serial D=256 reduction per
   block — fine at this width; the split-K rmsnorm exists for N=5120.
   A/B (env-toggled, 2 runs each): d512 B=1 **54.4/54.2 vs 53.3/53.5**, d2048
   B=1 **51.3/51.3 vs 50.4/50.6**; B=8 unchanged (141.7/143.0 vs 141.1/142.1).
   → **+2%**. Copied from agent-infer decode_prep.
2. **Grouped fp8 GEMV** (`kernels_linear.make_linear_fp8_gemv`, GROUP=4): the
   flat loop kept one 128-bit load in flight per thread. Load GROUP chunks
   first, then FMA; apply the block scale to the chunk partial, not per
   element. `scripts/ab_fp8_gemv.py`, 100 iters: 16480×5120 **44.7 → 41.0 µs**
   (63% roofline), 5120×6144 28.7 → 28.3 (unchanged; 33% roofline —
   launch/occupancy-bound at small N, the next fp8 target).

Together: d512 B=1 **19.28 → 18.38 ms/tick, 51.9 → 54.4 tok/s (+4.8%)**.

## Rule

Fusion of small kernels is worth ~2% inside the graph — take it when it also
removes copies, but the tick is 74% GEMV. Next: the fp8 out_proj shape at 33%
roofline (split-K across blocks), the B=8 WGMMA path at 3–4× the M=1 GEMV.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-27 | see bench-baseline.json | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | 0.639 | 18.45 (B=1) | 54.2 B=1 / 142.7 agg B=8; 41.5 @8k, 23.9 @32k |

Raw: `/work/bench_gpu.log`, A/B task logs (attn_prep on/off ×2, fp8 gemv).
