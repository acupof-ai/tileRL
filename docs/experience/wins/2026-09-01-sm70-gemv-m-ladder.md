# The sm70 M-row GEMV has no weight reuse; round M up a ladder — 2026-09-01

> Status: Shipped (the ladder). Speculation still rejected — see the verdict.

## Context

A speculative verify replay cost 271 ms at every width W>1 while a plain decode
replay cost 40.9 ms. Per-kernel profiling put 252 of those 271 ms in
`linear_fp4_gemv_sm70_m`, at 507.5 µs/call against the M=1 kernel's 64.5 µs.

## What Worked

The kernel's premise — load and decode each weight tile once, reuse it across M
rows — does not hold on this hardware. Compile-time M against cost at
17408x5120:

| M | µs | µs/row |
|---:|---:|---:|
| 1 | 127.6 | 127.6 |
| 2 | 244.8 | 122.4 |
| 4 | 508.6 | 127.2 |
| 8 | 1015.6 | 126.9 |
| 16 | 2010.5 | 125.7 |

**127 µs/row, flat from M=1 to M=16.** Exactly linear: the kernel is M serial
single-row passes wearing one launch.

ncu says why, identically at every shape: **255 registers/thread** (the hard
cap), **12.2% occupancy**, **DRAM 6.8%** against 42% SM throughput. The weight
bytes ARE being reused — l1tex hit rate is 88% — but bytes were never the
limiter. Issue bandwidth is, and an extra row is an extra row's worth of
instructions with no slack to hide them in.

So the fix is not to make reuse work; it is to stop paying for rows nobody
reads. The dispatch padded every M<=8 call up to 8. Now it rounds up a ladder
(M=2/4/8, registered variants of the same factory) and pads only to the next
rung:

| W | replay before | after | |
|---:|---:|---:|---|
| 1 | 40.9 ms | 40.9 ms | unchanged (different kernel) |
| 2 | 271.0 ms | **78.4 ms** | 3.5× |
| 4 | 271.7 ms | **146.8 ms** | 1.85× |
| 8 | 271.6 ms | 271.5 ms | unchanged (already the top rung) |

Cost is now linear in W instead of a step to full price at W=2. This also
covers B=2..4 batch serving, not just speculation.

Parity at the real projections (17408x5120, 12288x5120, 5120x17408) for
M=1,2,3,4,5,8 — including the odd widths that exercise the round-up — worst
1.40e-03 against the 1e-2 gate. 146 passed, 4 skipped.

## Verdict on speculation: still rejected

At 127 µs/row x 497 calls, **one extra verify row costs ~63 ms while a whole
dense decode tick costs 40.9 ms.** Verifying a token is more expensive than
decoding it, so the premise speculation rests on is false here:

| config | tick | tokens | tok/s |
|---|---:|---:|---:|
| dense | 40.9 | 1.00 | 24.4 |
| depth 1 (W=2) | 83.4 | 1.62 | 19.4 |
| depth 3 (W=4) | 161.7 | 2.24 | 13.9 |

The MTP head is fine (62% top-1 agreement, 97-100% accept in serving). The
economics are not, and they will not be until the GEMV's occupancy problem is
solved — which is a consequence of fp4 needing software decode on a card with
no fp4 tensor cores.

## Rule

"Reuse across M rows" is a claim about the limiter, not about the code. This
kernel really does reuse its weight tile (88% l1 hit) and still costs full price
per row, because it was issue-bound the whole time. Before optimizing for reuse,
check which resource is actually saturated — here DRAM sat at 6.8% while the
kernel was pinned at 255 registers.
