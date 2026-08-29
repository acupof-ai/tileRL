# mma8's bandwidth deficit is registers, not the load pattern — 2026-08-29

> Status: root cause measured. Shared-memory staging built, refuted, reverted.

## Context

`linear_*_mma8` is the decode GEMM for 2 <= M <= 8, and after the M-row GEMV
landed it still owns every decode with M >= 4 — which is every batched serve.
It moves the same weight bytes at 2.4-3.0x the GEMV's time, on EVERY shape in
the model (kernel time from the CUDA profiler, not wall time):

| N | K | MB | blocks | gemv M=1 | mma8 M=8 | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 5120 | 26.2 | 256 | 13.8 us | 38.8 us | 2.82 |
| 5120 | 6144 | 19.7 | 160 | 11.0 | 33.3 | 3.01 |
| 17408 | 5120 | 55.7 | 544 | 28.5 | 69.4 | 2.44 |
| 5120 | 17408 | 55.7 | 160 | 28.6 | 82.2 | 2.87 |
| 10240 | 5120 | 32.8 | 320 | 16.7 | 50.3 | 3.01 |
| 6144 | 5120 | 19.7 | 192 | 11.0 | 29.7 | 2.70 |

Flat across a 3.4x span of block counts, so it is not occupancy-through-grid.

## The hypothesis that was wrong

The mma B fragment wants 4 bytes per lane. As a global load that is 8
scattered 16-byte runs per warp, against the GEMV's 256 contiguous bytes — a
textbook coalescing argument, and Marlin (which this kernel's docstring cites)
stages W through shared memory precisely to avoid it.

Built it: 16 bytes per lane so the warp reads 8 rows x 64 CONTIGUOUS bytes into
shared, `[chunk_u32][row]` layout so the fragment read is bank-conflict-free.

**No effect. 69.2 us against 69.6.** Reverted.

(The one apparent win, G=8 at 61.5 us, was a wrong kernel: the staging loop
assumes 32 lanes x 16 B covers 8 rows x G*16 bytes, which holds only at G=4.
At G=8 it loaded half the weights. Its relative error was 1.20 against 2.55e-03
for the correct settings — the parity column in the probe is what caught it,
and it would have shipped as a 13% win without one.)

## The actual cause

ncu, same shape, same run:

| | gemv | mma8 |
|---|---:|---:|
| registers / thread | **64** | **128** |
| warps active (occupancy) | **45.2%** | **21.8%** |
| DRAM throughput | 44.7% | 18.5% |
| issue active | 59.3% | 44.6% |

128 registers per thread halves the resident warps, and DRAM throughput tracks
occupancy almost exactly (2.07x occupancy ratio, 2.42x bandwidth ratio). The
kernel is not short of bytes or of issue slots; it is short of warps to hide
the latency of the bytes it asks for.

The registers are in the per-iteration live set: `acc[NG*4]` (16) +
`w[G][NG]` (16) + `s[G][NG]` (16) + `xa[G]` (16 as 4 uint4), before decode
temporaries.

## The knobs cannot pay for it

`NG` (output rows per block) and `G` (chunks per iteration) are what size the
live set. Shrinking either makes it slower, because a smaller tile re-streams X
and does less mma work per weight byte (17408x5120, M=8, all rel 2.55e-03):

| NG | KW | G | us | GB/s | blocks |
|---:|---:|---:|---:|---:|---:|
| **4** | **4** | **4** | **69.6** | **800** | 544 |
| 4 | 4 | 2 | 72.1 | 773 | 544 |
| 2 | 4 | 4 | 96.1 | 580 | 1088 |
| 1 | 4 | 4 | 84.6 | 658 | 2176 |
| 2 | 2 | 4 | 91.2 | 611 | 1088 |
| 2 | 4 | 2 | 107.2 | 520 | 1088 |

The shipped configuration is the best of them. Relieving the register pressure
needs a different schedule, not a different tile size — parked here, with the
counters recorded so the next attempt starts from the binding resource.

## Rule

A coalescing argument is a hypothesis, not a diagnosis. Two of this session's
kernel hypotheses (GDN owns the verify cost; mma8's scattered loads) were
plausible, cheap to argue, and wrong — and both cost a build before ncu was
asked. Read the occupancy and DRAM counters FIRST when a kernel moves known
bytes in known time; the ratio between them names the binding resource in one
run.

Corollary: every A/B probe carries a correctness column. This one's would have
shipped a 13% "win" that computed half the matrix.
