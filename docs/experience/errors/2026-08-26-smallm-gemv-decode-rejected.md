# small-M GEMV for B=2..8 decode — REJECTED (1.56-1.61x slower than the padded WGMMA path), 2026-08-26

> Status: Killed — kernel/gates/tests reverted (no half-states; the code
> lives at 26e6471 in git history). Shipped: the write_tokens ABI fix
> (e2ea273), this entry, and the A/B harness (scripts/ab_smallm_gemv.py).

## Context

Hypothesis 2 of the decode Phase-2 recon
(`docs/experience/wins/2026-08-26-decode-tick-profile.md`): the GEMV dispatch
gate is `M == 1`, so B=8 decode runs the prefill WGMMA kernels — M padded
8->16 (50% wasted rows), k_split=2 atomics, an e4m3 A-quant launch per
linear — at 1.9-3.1x the B=1 GEMV's per-byte cost. The proposed lever:
generalize the fp4/bf16/fp8 GEMV from M=1 to a fixed M=8 (stream W once,
M-way FMA, no padding, no k_split, no A quant) and route 2<=M<=8 decode
through it. Gate: slice4 B=8 decode graph aggregate tok/s vs shipped, same
process, 30-tick steady-state avg, win = graph gain >= 3% AND relerr vs
shipped <= 1e-2.

Implementation: `make_linear_*_gemv_smallm` (kernels_linear.py, one warp per
output row, WQ streamed once, the warp's K-slice FMAed against all 8
activation rows, 8 warp reductions), the three backend gates behind
`_SMALLM_GEMV`, sm90-only registration. A/B harness:
`scripts/ab_smallm_gemv.py` (control = shipped gate, candidate = small-M,
same engine/model/inputs, B=1/2/4/8). H20 GPU 7 quiet-gated, commit
e2ea273.

## Root Cause

The candidate is **slower at every B>=2** (graph, 30-tick avg, same
process):

| B | control ms/tick | candidate ms/tick | candidate agg tok/s | delta |
|---|---:|---:|---:|---:|
| 1 | 1.7448 | 1.7473 | 572.3 | -0.1% (noise; M=1 path unchanged) |
| 2 | 3.9280 | 6.3311 | 315.9 | **-38.0%** |
| 4 | 4.1093 | 6.5642 | 609.4 | **-37.4%** |
| 8 | 4.5152 | 7.0558 | 1133.8 | **-36.0%** |

The hypothesis assumed the M=8 kernel would run at the B=1 GEMV's
bandwidth-bound efficiency (~49% of HBM roof, W streamed once). It does
not, for one structural reason:

**Every warp reloads the whole activation matrix.** Each warp owns one
output row and loads `X[m, base+v]` for all 8 m-rows every K-tile, so the
kernel's X traffic is `N_warps x 8 x K x 2` bytes — for lm_head
(N=248320, K=5120) that is **19.9 GB** of X loads (L2-cached, but
load-instruction/L1-bandwidth bound). The WGMMA path loads X once per
output block (16 rows shared across 64 N-columns): 636 MB — **31x less**.
On top of that, the candidate does 8 scalar FMAs per W element in the FP
pipe (10.2 GFLOP for lm_head) where WGMMA uses tensor cores, and the
serial `for m` loop has no software pipelining. Net: the candidate is
~2.3x slower than the B=1 GEMV per W byte, and 1.56-1.61x slower than the
padded WGMMA path it was meant to replace. The WGMMA path's sins (50%
padding, k_split atomics, A-quant launches) are cheaper than the GEMV's
per-warp X reload at M=8.

**Correctness is not the problem — the kernel is bit-identical to the
shipping M=1 GEMV.** Row-by-row diagnostic
(`scripts/_diag_smallm_parity.py`, fp4 K=288/5120/17408, fp8 K=1024/5120):
m8 output vs per-row M=1 GEMV output is max-abs 0.0, fro 0.0 at every
shape. The harness's `parity_vs_f32_ref: False` flags are a gate artifact,
not a kernel defect: both kernels fail `allclose(rtol=1e-2, atol=1e-2)`
against the f32 reference at model-scale K (fro 0.23-0.24%, concentrated
at near-zero output elements where the 1e-2 absolute tolerance dominates)
— the shipping M=1 kernel's own parity test only covers K<=128, so this
was never measured before. The 3.8% fro-relerr vs shipped is the *shipped
path's* e4m3 A-quant + requant noise: the candidate (bf16 A, no quant) is
closer to f32 truth than the shipped path is. Greedy tokens diverge at
B=2/4/8 because the shipped path's quant noise flips argmax decisions —
expected, and a point in the candidate's favor on accuracy, irrelevant
given the perf verdict.

A latent bug surfaced and was fixed along the way: the fused-qkv slice
`v` reaches `write_tokens` as a non-contiguous bf16 view (row stride = the
full qkv width). The f32 WGMMA path's bf16 cast copied it; the GEMV's
bf16 output made `_dev` a no-op, so the view violated the kernel's packed
ABI — tolerated at B=1 (the binder exempts size-1 dims), a hard crash at
B>=2. Fixed by `.contiguous()` at the `write_tokens` boundary (commit
e2ea273); no-op on every already-packed path.

## Fix

None for the kernel as scheduled — reverted (26e6471 in git history
holds the impl; it is correct, bit-identical row-by-row, so a
shared-memory-X variant starts from there, not from scratch). The next
lever, if the B=8 lever is
revisited: stage X in shared memory per block (X is 8xK = 80 KB — tile K,
load `X[8, K_tile]` once per block, all warps read it from shared), which
cuts the X traffic ~31x to WGMMA-comparable levels. Whether a
shared-memory-X GEMV then beats the WGMMA path's tensor cores at M=16 is
unsettled — the WGMMA path runs at 58% of roof and the padding waste is
rows, not time. Cheaper alternatives to try first: accept the WGMMA path
for B>=2 and attack its k_split=2 atomics / A-quant launch count instead.

## Rule

A small-M GEMV only wins on W bandwidth if X is staged once per block —
letting every warp reload the full M-row activation matrix makes X
traffic `N_warps x M x K`, which for lm_head-shaped N is 31x the WGMMA
path and swamps the W savings. And: a GEMV parity gate at K<=128 says
nothing about model-scale K — the bf16-IO GEMVs fail a strict
allclose(atol=1e-2) at K=5120+ on near-zero outputs; gate GEMV parity on
fro-relerr, not per-element allclose, past K~1024.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-26 | e2ea273 | H20 pod | cuda/sm90 | slice4 B=2 graph, control | — | 3.9280 | 509.2 agg |
| 2026-08-26 | e2ea273 | H20 pod | cuda/sm90 | slice4 B=2 graph, small-M GEMV | — | 6.3311 | 315.9 agg |
| 2026-08-26 | e2ea273 | H20 pod | cuda/sm90 | slice4 B=4 graph, control | — | 4.1093 | 973.4 agg |
| 2026-08-26 | e2ea273 | H20 pod | cuda/sm90 | slice4 B=4 graph, small-M GEMV | — | 6.5642 | 609.4 agg |
| 2026-08-26 | e2ea273 | H20 pod | cuda/sm90 | slice4 B=8 graph, control | — | 4.5152 | 1771.8 agg |
| 2026-08-26 | e2ea273 | H20 pod | cuda/sm90 | slice4 B=8 graph, small-M GEMV | — | 7.0558 | 1133.8 agg |

Raw artifacts: `scripts/ab_smallm_gemv.py` JSON stdout (BENCH_COMMIT=e2ea273,
GPU 7 quiet-gated, 30-tick avgs, same process);
`scripts/_diag_smallm_parity.py` stdout (m8 vs M=1 bit-identity). Dev
tooling exempt from the bench-entry rule.

## Iteration

Wall time ~2.5 h, 4 pod round-trips: (1) first A/B run — candidate crashed
at B=2 on the write_tokens packed-ABI violation; (2) root-caused to the
bf16 view surviving `_dev`'s no-op cast, fixed with `.contiguous()`,
re-ran — full A/B, candidate 1.56-1.61x slower at B>=2; (3) parity
diagnostic — m8 vs M=1 GEMV bit-identical, the parity flags are a
gate artifact both shipping kernels trip at model-scale K; (4) this entry.
The implementation itself was pre-seeded in the worktree (kernels + gates
+ harness, commit 26e6471); this arm's work was the A/B, the ABI fix, and
the verdict.
