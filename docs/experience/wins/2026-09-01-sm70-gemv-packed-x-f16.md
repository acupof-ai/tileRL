# Pre-packing X to f16 broke the sm70 GEMV's per-row floor — 2026-09-01

> Status: Shipped. Supersedes the "127 us/row flat" finding in
> `2026-09-01-sm70-gemv-m-ladder.md` — the ladder stays, its ceiling does not.

## Context

The M-row GEMV cost 127 us/ROW flat from M=1 to M=16, so batching bought
nothing and a speculative verify row cost more than a whole decode tick. ncu
said the kernel was issue-bound (255 regs/thread, 12.2% occupancy, DRAM 6.8%
against 42% SM throughput), which read as a hardware wall.

It was not the hardware. It was X.

## What Worked

Two measurements on the same kernel, both about the activation and neither
about the weights:

- **Traffic.** X is re-read by every block. At M=8, N=17408, K=5120 that is
  `8 rows x 5120 f32 x 4352 blocks = 0.71 GB` = 792 us at 900 GB/s, **78% of
  the measured 1015.6 us**. The weight side was already healthy — M=1 hits 349
  GB/s effective at an 88% l1tex hit rate.
- **Instructions.** 32 of the ~49 per-row instructions were converting X from
  f32 to f16 *inside the tile loop* — 16 `__float2half_rn`, 8 shifts, 8 ors,
  re-done for every tile of every block.

Both disappear if X arrives already packed as f16. The kernel gained an `xh`
flag (`tl_fp4_gemv_tiles_f16_m_xh`: two `ld.global.nc.v4.u32` where there were
four `v4.f32` plus the convert block), and the dispatch site packs once.

Measured, 3 warmup + 20 timed:

| shape | M=1 | M=2 /row | M=4 /row | M=8 /row |
|---|---:|---:|---:|---:|
| 17408x5120 | 128.1 -> **88.2** | 120.2 -> **44.9** | 126.9 -> **33.8** | 127.5 -> **34.0** |
| 12288x5120 | 92.5 -> **82.4** | 87.4 -> **33.6** | 90.8 -> **24.8** | 90.7 -> **24.2** |
| 5120x17408 | 133.9 -> **93.4** | 129.1 -> **56.7** | 133.0 -> **37.1** | 122.9 -> **37.2** |

**Bit-exact** (relerr 0.00e+00 at every shape and M): both paths round to
nearest f16, the packing just does it once instead of per tile. Parity through
the real dispatch at the three projection shapes, M=1,2,3,4,5,8: worst
4.93e-04 against the 1e-2 gate.

The per-row cost is no longer flat — 34 us/row at M=8 vs 88 at M=1. **Batching
finally amortizes**, which is the whole premise verification rests on.

M=1 joined the ladder (1.12-1.45x over the scalar GEMV it replaced), so the
f32 rungs have no caller left and are gone; the ladder is f16-X only.

## Effect on the tick

Verify replay, B=1, torch profiler per kernel:

| W | before | after | marginal row |
|---:|---:|---:|---:|
| 1 | 38.6 ms | **30.9 ms** | — |
| 2 | 41.6 ms | 41.6 ms | +10.7 ms |
| 4 | 55.7 ms | 55.7 ms | +14.1 ms |

**One verify row now costs 10.7 ms against a 30.9 ms dense token.** It was 63
ms. Verification is finally cheaper than decoding, which is the condition
speculation needs and never had on this card.

End-to-end dense B=1 through the server (two-point slope, 32 vs 288 tokens):

| ctx | before | after |
|---:|---:|---:|
| 31 | 25.8 | **32.7 tok/s** |
| 1052 | 23.1 | **27.4 tok/s** |

## Speculation now ships

With a verify row at 10.7 ms against a 30.9 ms token, depth 3 pays:

| workload | dense | spec depth 3 | |
|---|---:|---:|---:|
| counting (control) | 32.7 | **52.7 tok/s** | 1.61x |
| coding | 32.6 | **43.4 tok/s** | 1.33x |
| dialogue | 31.9 | 32.0 | 1.00x |
| thinking | 32.0 | 30.2 | 0.94x |

52.7 is 82% of the 64 tok/s weight roofline, but counting is near-zero-entropy
under greedy decode — coding at 1.33x is the honest headline. A first reading
of 1.3 tok/s was a warmup artifact in the bench, not the engine
(`errors/2026-09-01-spec-warmup-one-width.md`), and the depth that gets this is
exactly 3 — depth 4 spills the sm70 verify ladder and loses to dense
(`errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md`).

## Rule

"Issue-bound" names the resource, not the cause. ncu said 255 regs and 12.2%
occupancy, and that was read as a hardware ceiling for two rounds of work; the
instructions being issued were mostly a format conversion that did not need to
be in the loop at all. When a profiler says a kernel is issue-bound, count what
the instructions ARE before concluding the shape is wrong — and check the
activation, not just the weights, when the operand you keep re-reading is the
small one.
