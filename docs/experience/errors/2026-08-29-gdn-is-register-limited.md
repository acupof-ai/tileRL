# The GDN prefill kernel is register-limited: 255/thread, 2 blocks/SM, 6.25% occupancy

## Context

The kernel's per-step cost is 3.05 us and flat across T = 64..512. By FMA count
it should be ~0.27 us. That 11x was unexplained all day, and four kernel
variants were written against guesses about it: three chunkwise-WY rewrites
(all slower), an epilogue move (+6.1%, shipped), and a V split (wrong and
slower). None of them measured where the time went.

## The Measurement

One `ncu` run answers it (`--section WarpStateStats --section Occupancy
--section SpeedOfLight`, via `scripts/occ_gdn.py`):

| | |
|---|---:|
| Registers per thread | **255** (the hardware cap) |
| Block Limit Registers | **2** |
| Block Limit Shared Mem | 6 |
| Theoretical occupancy | 12.5% |
| Achieved occupancy | **6.25%** — 4 warps of a possible 64 |
| Compute (SM) throughput | 11.57% |
| DRAM throughput | **0.28%** |
| ncu's own estimate | Est. Local Speedup 87.5% |

Not bandwidth. Not arithmetic. **Registers**, and they are at the cap, which
also means the compiler is spilling.

The dominant consumer is `state_local`: the per-thread state column, K=128
floats, i.e. 128 of the 255 registers. It comes from a shipped win — "the state
column lives in a per-thread local array (registers/L1) ... 21.6% faster than
the global-state f32 baseline". That optimisation traded registers for memory
traffic and won on its own A/B; **occupancy was not measured, so the price it
paid stayed invisible for four days.**

## What This Explains

- **Why three chunkwise-WY rewrites lost.** They reduce arithmetic. Arithmetic
  is 11.57% busy; the machine is idle waiting on a warp count of 4.
- **Why the V split was wrong AND slower.** More blocks was the right
  direction, but slicing V does not reduce registers per thread — every thread
  still carries a full 128-element state column — so residency did not move,
  and the split duplicated the q/k half on top.

## The Lever It Names

Halve `state_local`. Splitting the state column across two threads (a K split,
not a V split) takes it to 64 floats each, roughly doubling resident blocks,
and costs one 2-thread shuffle reduction in the `k·S` and `q·S` contractions.
Not attempted here.

## Rule

An optimisation that trades one resource for another has to be measured on
both. "Local state column: 21.6% faster" was true and shipped, and it bought
that with the occupancy that now caps the kernel at 6.25%. When a win's
mechanism is "keep it in registers", read the occupancy afterwards.

And: one `ncu` run cost less than any of the four variants written against
guesses, and it is the only thing all day that explained the number instead of
moving it.
