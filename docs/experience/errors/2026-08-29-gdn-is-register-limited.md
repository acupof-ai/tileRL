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

## Three probes, all negative — what the cheap paths cannot do

Before writing a K split against the register hypothesis, three experiments
tested it. All cost minutes; the K split would have cost an hour.

| probe | result |
|---|---|
| halve `state_local` (K=128 -> 64 floats, wrong math, real registers) | Block Limit Registers **still 2**, occupancy **still 6.25%** |
| `tl.ptxas_register_usage_level` = 0, 2, 5, 8 | occupancy **6.25%** at every level; us/step 3.00-3.07 |
| the same half-state probe's speed | 3.05 -> 2.52 us/step, **-17%** |

The first two kill the cheap fixes: the 255 registers are not one array's size,
and they are not ptxas being permissive. Whatever holds them there is not
reachable by shrinking a buffer or turning a knob.

The third is the useful one, and it argues against the diagnosis being complete:
if the kernel were purely occupancy-bound, removing half the arithmetic would
buy nothing, and it bought 17%. So the cost is **split** between the 4-warp
residency and the per-token dependency chain, and fixing either alone caps out
well below ncu's 87.5% estimate.

A K split — the state column across two threads — remains untested, but its
premise is now weaker than it looked: halving that array did not move the
register limit, so splitting it across threads may not either.

## Rule, second half

Three cheap probes against one hypothesis beat one expensive implementation of
it. Two of these were negative in the useful way — they deleted a candidate —
and the third quietly refuted the framing by showing arithmetic still matters.
None of that would have surfaced from writing the K split and measuring the
result, which would have said only "slower" and left the reason open, exactly
as the three chunkwise rewrites did.
