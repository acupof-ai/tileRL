# Batched-decode arms (shared-X small-M GEMV + k_split=1 WGMMA) — REJECTED, 2026-08-26

> Status: Killed — kernel/gates/flags reverted (no half-states; the code
> lives at 0299f78 / 0518d76 in git history). Shipped: this entry, the A/B
> harness (`scripts/ab_batch_decode.py`), and the launch script
> (`scripts/_ab_batch_launch.sh`) as dev tooling.

## Context

The follow-up to the rejected small-M GEMV arm
(`2026-08-26-smallm-gemv-decode-rejected.md`). That arm's kernel was correct
(bit-identical row-by-row) but 1.56-1.61x slower because every warp reloaded
the full 8-row activation (31x the WGMMA path's X traffic). Its errors entry
named the fix (stage X in shared memory once per block) and the open question
(whether a shared-X GEMV then beats the WGMMA path's tensor cores at M=16),
plus a cheaper alternative (attack the WGMMA path's k_split=2 atomics
instead). This arm tested both, on the same harness method: slice4 decode
graph ON, steady-state 30-tick avg, same process, control = shipped. Win
gate: B=8 aggregate tok/s gain >= 3% AND fro-relerr vs shipped <= 1e-2,
B=1 neutral (the M=1 path is settled at 55.4% roof).

- **H1 (primary): shared-X small-M GEMV.** The rejected arm's kernel
  (one warp per output row, WQ streamed once, M-way FMA, warp-LUT fast
  decode) with XQ staged to shared once per K-tile per block (all warps read
  it from shared). A is e4m3 (same per-token quant as shipped) so the output
  matches shipped within summation noise — the previous arm's bf16 A was
  3.8% fro off shipped and would have failed the relerr gate.
- **H2 (cheaper): k_split=1 WGMMA for 2<=M<=8.** The fp4 prefill path runs
  `linear_fp4_fp8` with k_split=2 (f32 atomic adds into a zeroed output).
  At decode M=8 (padded to 16) the split's 2x blocks + atomics look like
  pure cost; H2 registers a k_split=1 cell and gates it on for decode.

## Root Cause

Both candidates are **slower at B=8** (graph, 30-tick avg, same process):

| arm | B=1 ms | B=2 ms | B=4 ms | B=8 ms | B=8 agg tok/s | B=8 delta |
|---|---:|---:|---:|---:|---:|---:|
| shipped | 1.7493 | 3.9356 | 4.1181 | 4.4806 | 1785.5 | — |
| ks1 (all N) | 1.7498 | 4.3510 | 4.5309 | 4.9001 | 1632.6 | **-9.4%** |
| smallm (shared-X) | 1.7510 | 9.4175 | 9.5307 | 9.7587 | 819.8 | **-118% (2.18x slower)** |

Second run, ks1 gated to large-N only (N>=10240, where the grid is already
>= 2 waves without the split):

| arm | B=1 ms | B=2 ms | B=4 ms | B=8 ms | B=8 agg tok/s | B=8 delta |
|---|---:|---:|---:|---:|---:|---:|
| shipped | 1.7526 | 3.9585 | 4.1198 | 4.4652 | 1791.6 | — |
| ks1 (N>=10240) | 1.7527 | 4.0547 | 4.2281 | 4.5839 | 1745.2 | **-2.6%** |

**H1 (smallm) — the binding constraint is the scalar FMA pipe, not X
traffic.** The shared-X fix did what it claimed: XQ is staged once per
K-tile per block (2 KB e4m3), all warps read it from shared — the 31x X
traffic is gone. But the kernel is still 2.18x slower than shipped, and
*slower than the rejected arm's per-warp-reload kernel* (9.76 vs 7.06 ms).
The reason: at M=8 each warp does 8 scalar FMAs per W element (85
inst/micro-tile vs the M=1 kernel's 29), and the WGMMA path does the same
work on tensor cores. The X reload was a real cost but never the binding
one — the 8x FMA multiplicity is. The shared-X barrier + e4m3 quant/casts
added overhead without touching the FMA bottleneck, so the net regressed.
The previous errors entry's open question is settled: **a shared-memory-X
GEMV does not beat the WGMMA path's tensor cores at M=16.** The GEMV's
W-bandwidth advantage (stream W once at ~49% roof) only exists at M=1,
where there is no FMA multiplicity; at M=8 the scalar FMA issue pressure
(8 independent chains, but 8x the instructions) swamps it.

**H2 (ks1) — k_split=2's atomics are not pure cost; the split is a
dequant-parallelism optimization.** Removing the split lost 9.4% (all N)
and even the large-N-only gate lost 2.6%. The fp4 WGMMA path's bottleneck
is the e2m1fn->e4m3 dequant (the vectorized shared-memory macro), not the
WGMMA. k_split=2 puts two SMs on the same output tile's K-range
concurrently, so the tile's dequant finishes in half the wall time — the
atomic adds are cheaper than the dequant latency they hide. This holds
even where the grid is over-saturated (lm_head N=248320): the split is
per-tile dequant parallelism, not grid occupancy. The recon's "k_split=2
atomics are pure cost at M=16" was wrong — it read the split as an
occupancy play, but it is a dequant-latency play.

**Correctness is not the problem for either arm.** ks1 is bit-identical to
shipped (fro-relerr 0.0 at every shape; same math, no atomics — the split
sum and the single accumulator differ only in floating-point
associativity, here exactly zero because the K-tile count per split is
even and the partial sums are identical). Greedy tokens are identical at
B=1/2/4/8. smallm (e4m3 A, same quant as shipped) is sub-1e-2 vs shipped
by construction (same A-quant; the only difference is W dequant to fp32
vs shipped's e4m3 requant, which averages down over K) — not measured
directly, because the arm was dropped after the 2.18x perf loss, but the
e4m3-A choice (vs the previous arm's bf16 A, which failed the relerr gate
at 3.8% fro) made it a perf-only rejection with no correctness question
to untangle. B=1 is neutral for every arm (+0.01% drift; the M=1 GEMV
path is untouched).

## Fix

None for either arm — reverted (0299f78 / 0518d76 hold the impls; both are
correct, so a future revisit starts from there, not from scratch). The
shipped WGMMA path (M padded 8->16, k_split=2, e4m3 A-quant) is the right
kernel for B=2..8 decode. **The B=8 lever (recon hypothesis 2) is dead.**
The remaining decode levers are the B=1 GEMV dequant issue throughput
(hypothesis 1, the 80 tok/s lever) and batched greedy sampling (hypothesis
3, the B-scaled leak) — neither is helped by this arm's findings.

## Rule

1. A small-M GEMV cannot beat the WGMMA path at M>=8 even with X staged to
   shared: the binding constraint is the Mx scalar-FMA issue pressure (85
   inst/micro-tile at M=8), not X traffic. The GEMV's W-bandwidth edge
   exists only at M=1 (no FMA multiplicity). Do not revisit the small-M
   GEMV for decode; the WGMMA path is settled for B>=2.
2. k_split=2's atomics are not pure cost at small M on the fp4 WGMMA path:
   the split doubles per-tile dequant parallelism (two SMs on the same
   tile's K-range), and the dequant is the bottleneck. Gate k_split=1 only
   where the dequant is not on the critical path (i.e., not the fp4 path
   at small M) — and measure, don't assume the atomics dominate.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-26 | 0299f78 | H20 pod | cuda/sm90 | slice4 B=8 graph, shipped | — | 4.4806 | 1785.5 agg |
| 2026-08-26 | 0299f78 | H20 pod | cuda/sm90 | slice4 B=8 graph, ks1-all | — | 4.9001 | 1632.6 agg |
| 2026-08-26 | 0299f78 | H20 pod | cuda/sm90 | slice4 B=8 graph, smallm | — | 9.7587 | 819.8 agg |
| 2026-08-26 | 0518d76 | H20 pod | cuda/sm90 | slice4 B=8 graph, shipped | — | 4.4652 | 1791.6 agg |
| 2026-08-26 | 0518d76 | H20 pod | cuda/sm90 | slice4 B=8 graph, ks1 N>=10240 | — | 4.5839 | 1745.2 agg |

Raw artifacts: `scripts/ab_batch_decode.py` JSON stdout (BENCH_COMMIT
0299f78 / 0518d76, GPU 6 quiet-gated, 30-tick avgs, same process);
`/work/ab_batch.log` on the pod. Dev tooling exempt from the bench-entry
rule.

## Iteration

Wall time ~1.5 h, 2 pod round-trips: (1) implemented both arms behind
flags + the 3-arm harness, first A/B — smallm 2.18x slower (dead), ks1-all
-9.4% (the small-N linears lost their 2nd wave); (2) gated ks1 to large-N
only + fixed the harness's GPU/CPU relerr device mismatch, re-ran — ks1
-largeN -2.6% (the split is dequant parallelism, not occupancy), reverted,
this entry. The smallm kernel's e4m3-A choice (vs the previous arm's bf16)
was decided upfront from the relerr gate and was correct: it made the
rejection perf-only (fro-relerr 0.0 vs shipped), with no correctness
question to untangle.
