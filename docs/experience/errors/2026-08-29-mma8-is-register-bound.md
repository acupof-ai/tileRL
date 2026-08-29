# mma8 moves bytes at 40% of the GEMV's rate — 1.93x the load instructions, not registers — 2026-08-29

> Status: cause established on the fourth attempt — mma8 issues **1.93x the
> load instructions for identical DRAM traffic**. The "register-bound" title
> this entry shipped with was inferred from a correlation and is retracted
> below; three fixes built on it all failed, which is the point of the entry.

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

## Third refutation: the register cap is already saturated

`__launch_bounds__(128, N)` injected into the generated source via tilelang's
`register_cuda_postproc` (one process and one cache dir per arm — the postproc
is not part of the JIT cache key, so a second arm in the same process silently
reuses the first's binary):

| blocks/SM forced | us | GB/s | rel |
|---:|---:|---:|---|
| as emitted | 69.5 | 802 | 2.55e-03 |
| 2 | 69.5 | 802 | 2.55e-03 |
| 3 | 69.5 | 802 | 2.55e-03 |
| 4 | 69.7 | 799 | 2.55e-03 |
| 6 | **80.4** | 693 | 2.55e-03 |

2, 3 and 4 are identical because ptxas **already fits 4 blocks per SM**:
128 threads x 128 registers = 16K of the SM's 64K, and 4 blocks x 4 warps = 16
warps of 64 is the 21.8% ncu reported. The injection works — arm 6 proves it,
by forcing registers under ~85 and paying 16% in spill.

So the kernel sits exactly at its register-determined occupancy ceiling, and
pushing past it loses. Three independent levers are now closed:

1. **NG / G tile knobs** — every setting below the shipped one is slower.
2. **Shared-memory weight staging** — 69.2 vs 69.6, nothing.
3. **The register cap** — saturated, and worse when forced past.

A Marlin-style pipeline (cp.async into shared, `ldmatrix` fragments,
double-buffered) is the next thing to try — with the rationale in "What it
actually is" below, cutting LSU requests, NOT the register story this entry
originally gave, which the NG sweep refutes. It would be the ONE lever that
moves both batched decode and speculation —
speculation's break-even needs the per-width cost under 6.0 ms against 8.9
today, and a decode GEMM flat in M is exactly what would deliver it
([wins/2026-08-29-spec-decode-net-win.md](../wins/2026-08-29-spec-decode-net-win.md)).

## The diagnosis does not survive

I named "register pressure -> occupancy -> bandwidth" from the correlation in
the table above (128 vs 64 registers, 21.8% vs 45.2% occupancy, 18.5% vs 44.7%
of DRAM peak). Sweeping NG with ncu breaks it:

| NG | grid | regs/thread | occupancy | DRAM | us |
|---:|---:|---:|---:|---:|---:|
| **4** (shipped) | 544 | 128 | 22.0% | 18.4% | **69.5** |
| 2 | 1088 | **200** | 12.2% | 13.6% | 96.1 |
| 1 | 2176 | 118 | 23.9% | 15.3% | 84.6 |

Two things break:

- **Shrinking the tile does not shrink the register count.** By the live set I
  counted (`acc[NG*4] + w[G][NG] + s[G][NG] + xa[G]`), NG=1 should need ~28
  registers. It uses 118, and NG=2 uses *more* than NG=4. So the 128 registers
  are not the arrays I attributed them to, and the whole "cut the live set"
  premise rests on a mis-reading.
- **At equal occupancy the big tile still wins.** NG=1 sits at 23.9% occupancy
  — slightly HIGHER than the shipped kernel's 22.0% — and is 22% slower. If
  occupancy were the binding resource that could not happen.

Together with the launch-bounds result (forcing higher occupancy is neutral to
worse), occupancy is not what separates mma8's 800 GB/s from the GEMV's 1950.

## What it actually is: twice the load instructions

The counters that name it, same shape, same run:

| | gemv | mma8 |
|---|---:|---:|
| `dram__bytes_read.sum` | 55.86 MB | 56.46 MB |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | **731,136** | **1,410,048** |

**Identical DRAM traffic, 1.93x the load requests.** So nothing is re-fetched
(L2 is fine, and the "it must be coalescing so it reads more" story is wrong in
its second half) — mma8 simply issues twice the load INSTRUCTIONS for the same
bytes, because the mma B fragment wants 4 bytes per lane where the GEMV's tile
takes 8. It is LSU-request-bound, and 1.93x on requests against 2.44x on time
is the right order.

This also explains why the shared-memory staging measured neutral rather than
refuting the access-pattern idea: it made the GLOBAL loads wide (16 B/lane) and
then added shared-memory loads to get the fragments back out, so the total
request count did not move. The implementation relocated the bottleneck instead
of removing it.

The fix therefore has to cut TOTAL load instructions, not widen them:
`ldmatrix` loads a whole 16x16 fragment in ONE instruction and distributes it
across the warp — which is the Marlin design, but for this reason rather than
the register one this entry originally gave.

## The other fp4 decode GEMM already in the tree loses too

`linear_fp4_fp8_decode` (fp4 -> e4m3 dequant into an fp8 WGMMA) is registered
for the decode bucket but unreachable, because the mma8 branch takes
`2 <= M <= _MX` first. Nobody had compared them at M=8. Setting `_MX = 7` makes
M=8 fall through, which is the whole A/B:

| N x K | mma8 | w4a8 | | rel mma8 | rel w4a8 |
|---|---:|---:|---:|---:|---:|
| 8192x5120 | 38.8 us | 49.6 | 1.28x slower | 2.8e-03 | 3.8e-02 |
| 5120x6144 | 33.3 | 44.4 | 1.33x | 2.2e-03 | 3.5e-02 |
| 17408x5120 | 67.0 | 88.9 | 1.33x | 2.2e-03 | 3.6e-02 |
| 5120x17408 | 76.9 | 106.3 | 1.38x | 2.3e-03 | 4.2e-02 |
| 10240x5120 | 46.6 | 60.1 | 1.29x | 2.3e-03 | 3.8e-02 |
| 6144x5120 | 27.7 | 39.1 | 1.41x | 2.1e-03 | 3.5e-02 |

Slower AND 16x less accurate (it quantizes the ACTIVATION to e4m3 as well).
The shipped dispatch is right.

## What is left, and what it is worth

The fp8 twin `tl_fp8_mma_rows` already loads `v2.u32` — 8 bytes per lane —
where the fp4 one loads `u32`. That is the same diagnosis from another angle:
fp8's mma8/gemv ratio is 2.07x against fp4's 2.64x.

So the remaining change is to widen fp4's weight load to 8 bytes, which needs
the lane -> k map re-cut (a lane would own 16 fp4 values spanning two chunks,
and the A fragment has to use the same permutation — the kernel already relies
on a "virtual k" permutation, so this is legal, just fiddly).

**Worth estimating before building, since three attempts on this kernel have
already failed:** halving the weight loads takes requests from 1.41M toward
~0.9M, so ~1.5x fewer, predicting the kernel at ~48 us against 69.5. That is
B=8 decode 311 -> ~400 tok/s (1.3x), and speculation's per-width cost 8.9 ->
~6.5 ms — **still above the 6.0 ms break-even**. A real win, not a
goal-flipping one.

## Rule

A correlation is not a diagnosis, and the counter that settles it is usually
one metric away. Occupancy and register count correlated beautifully with the
gap and were both wrong; `l1tex__t_requests` against `dram__bytes_read` — two
numbers, one ncu run — says it in a line. Three fixes were built on the
correlation before that run happened.

A coalescing argument is a hypothesis, not a diagnosis. Two of this session's
kernel hypotheses (GDN owns the verify cost; mma8's scattered loads) were
plausible, cheap to argue, and wrong — and both cost a build before ncu was
asked. Read the occupancy and DRAM counters FIRST when a kernel moves known
bytes in known time; the ratio between them names the binding resource in one
run.

Corollary: every A/B probe carries a correctness column. This one's would have
shipped a 13% "win" that computed half the matrix.
