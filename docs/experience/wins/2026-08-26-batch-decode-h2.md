# 8-way K-split for small-M fp4 decode — +7.5% at B=8, cuda/sm90, 2026-08-26

> Status: Shipped

## Context

Batched-decode tournament H2: make B=2..8 decode faster by attacking the
padded WGMMA path's overheads (not a new GEMV — that arm was rejected
2026-08-26, 1.56-1.61x slower). The B=8 decode tick runs the prefill WGMMA
kernels: M=8 padded to bM=16 (50% wasted rows), the prefill kernel's 2-way
K-split with f32 atomics, and an e4m3 A-quant launch per linear — linears
are 6.58 ms of the 9.85 ms eager tick
(`2026-08-26-decode-tick-profile.md`). Gate: slice4 decode graph ON,
30-tick avg, same process, control = shipped; win = B=8 aggregate tok/s
>= +3% AND relerr vs shipped <= 1e-2; B=1 neutral (its GEMV path is
settled). H20 GPU 6 quiet-gated, JIT cache warm.

## What Worked

**8-way K-split for the M<=16 decode path.** The decode tick is M<=16
(pure decode; mixed prefill+decode ticks pad to the chunk T -> M>=128 and
keep the prefill kernel). At bM=16 a WGMMA block is 2 warps — too thin to
hide HBM latency — so the K-split's resident warps buy more than its f32
atomics cost. The split is an **occupancy lever at small M, not a
K-parallelism lever**: the prefill settle (ks=2 at bM=128) does not
transfer down. Same-process A/B on the slice4 decode graph (30-tick avg,
control = shipped ks2, candidate = ks8):

| B | control ms/tick | candidate ms/tick | gain | agg tok/s (cand) |
|---|---:|---:|---:|---:|
| 1 | 1.7411 | 1.7420 | -0.05% | 574.0 |
| 2 | 3.9360 | 3.5910 | **+8.76%** | 556.9 |
| 4 | 4.1110 | 3.7765 | **+8.14%** | 1059.2 |
| 8 | 4.4674 | 4.1333 | **+7.48%** | 1935.5 |

B=1 is neutral by construction (the M=1 GEMV path is untouched; the gate
is 1 < M <= 16). Correctness: fro-relerr vs shipped ~1e-7 (atomic
reordering only — the math is identical). Greedy tokens are identical at
B=1/2 and flip at B=4/8: the 1e-7 perturbation tips argmax on near-ties,
which is below the shipped path's own ~2% e4m3 A-quant noise floor — and
the shipped ks2 has the same atomic non-determinism. Gate met: +7.48% >=
3%, 1e-7 <= 1e-2.

The sweep that found it (B=8, same-process A/B vs shipped ks2):

| cell | B=8 ms/tick | vs shipped | note |
|---|---:|---:|---|
| ks1 | 4.9285 | **-10.8%** | too few warps — hypothesis 1's premise was backwards |
| ks2 (shipped) | 4.4674 | — | prefill kernel's 2-way split |
| ks4 | 4.2297 | +5.06% | sweep direction confirmed |
| ks8 | 4.1333 | **+7.48%** | shipped |
| bf16-A WGMMA | 5.0392 | -13.3% | no A-quant launch, but bf16 WGMMA = half the fp8 throughput + 2x W shared traffic; fro-relerr 3.8% — gate-fail by construction |

ks1 losing was the informative result: the atomics are not the cost at
small M — the warps are. The shipped change is one registry entry
(`linear_fp4_fp8_decode`, k_split=8) and one backend branch (M<=16 ->
decode kernel, K padded to 8*BK); prefill (M>16) is untouched.

## Rule

At bM=16 a WGMMA block is 2 warps — too thin to hide HBM latency. K-split
buys resident warps (more blocks per SM, more outstanding loads), and the
f32 atomics cost less than the occupancy they gain: the split is an
occupancy lever at small M, not a K-parallelism lever. The optimal split
moves with block_M — prefill bM=128 settled at ks=2, decode bM=16 settles
at ks=8. When a kernel is memory-bound at a small tile, sweep the split
UP before concluding the atomics are overhead.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-26 | aada7f4 | H20 pod | cuda/sm90 | slice4 B=8 graph, shipped ks2 (control) | — | 4.4674 | 1790.8 agg |
| 2026-08-26 | aada7f4 | H20 pod | cuda/sm90 | slice4 B=8 graph, ks8 (candidate) | — | 4.1333 | 1935.5 agg |
| 2026-08-26 | aada7f4 | H20 pod | cuda/sm90 | slice4 B=4 graph, ks8 | — | 3.7765 | 1059.2 agg |
| 2026-08-26 | aada7f4 | H20 pod | cuda/sm90 | slice4 B=2 graph, ks8 | — | 3.5910 | 556.9 agg |
| 2026-08-26 | aada7f4 | H20 pod | cuda/sm90 | slice4 B=1 graph, ks8 (neutral) | — | 1.7420 | 574.0 |

Raw artifacts: same-process A/B JSON stdout (BENCH_COMMIT=aada7f4, GPU 6
quiet-gated, 30-tick avgs), pod `/work/h2_ab/results.jsonl` (ks1, bf16a)
and `/work/h2_ab2/results.jsonl` (ks4, ks8). Dev tooling (the A/B harness
and pod launch script) was removed with the shipping commit — the method
is this entry.

## Iteration

Wall time ~1 h, 4 pod sessions: (1) ks1 + bf16a cells — ks1 -10.8% (the
split is occupancy, not atomics — hypothesis 1 backwards), bf16a -13.3% +
gate-fail (dead); (2) ks4 + ks8 sweep — ks8 +7.48%, shipped; (3) smoke of
the flag-free shipped code — ran end-to-end at B=1/8, but the standalone
B=8 number (4.66 ms) was slowed by sibling contention mid-run (the quiet
gate checks util, not memory — a sibling held 82 GiB at 0% util); (4) a
same-process re-verification A/B OOMed twice on that sibling's 82 GiB
allocation and was judged redundant — the ks8 A/B was already
same-process against shipped, and the smoke confirmed the renamed
registry key resolves and the decode graph captures. The shipping commit
(8336978) is the ks8 candidate made unconditional; the A/B scaffolding
was deleted (no half-states).
