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

## It is not register-limited. It is SPILLING.

`ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum` on one
T=128 launch:

| | sectors |
|---|---:|
| local loads | 4,535,040 |
| local stores | 4,141,824 |
| global loads | 5,441,472 |

**Local-memory traffic exceeds global.** 8.7M sectors is ~278 MB of spill
traffic for a single launch. The per-thread state column is not in registers at
all — it is in local memory, which is global memory backed by L1.

This is the whole story, and it makes every earlier observation fall out:

- **The 11x against the FMA estimate**: each state access is a memory access,
  not a register access. The arithmetic was never the cost.
- **255 registers/thread**: ptxas spent the cap trying to hold it and still
  spilled.
- **Halving `state_local` changed nothing**: 64 floats spill as surely as 128
  once everything else is resident.
- **Why the shipped win still measured +21.6%**: local memory is L1-cached, so
  it genuinely beat the explicit-global baseline it was compared against. The
  comparison was to something worse, not to registers, and the name "local
  state column (registers/L1)" then made the spill invisible for four days.

## What Follows

Two directions, both now motivated by a measurement rather than a guess:

1. **Split the state column across threads** so each holds 32-64 floats and
   actually fits. Needs a 2- or 4-thread reduction for `k·S` and `q·S` through
   shared memory — warp primitives are off the table (AGENTS.md: block-parallel
   only).
2. **Put it in shared memory deliberately, with a padded layout.** A shared
   state tile was tried once and was 1.7x slower "LDS bank conflicts" — that is
   a layout problem, not a verdict on the approach, and 128 threads x 128 floats
   is 64 KB, which fits.

## Rule, third part

"In registers" is a claim about the generated code, not about the source. A
local array large enough to spill is a memory access with a register's syntax.
When an optimisation's mechanism is residency, verify it with the spill
counters — not with the A/B against whatever it replaced, which only proves it
beat that.

## Removing the spill made it 2.5x SLOWER

The state column moved to a `[K, V]` shared tile — thread `tv` owns column
`tv`, so for a fixed row the block reads consecutive addresses, one per bank.
I expected this to beat the earlier "shared state tile is 1.7x slower, LDS bank
conflicts" rejection, on the theory that it had used the transposed `[V, K]`
layout where every thread hits the same bank.

| | local (shipped) | shared [K,V] |
|---|---:|---:|
| parity | passes | **passes** |
| local-load sectors | 4,535,040 | **0** |
| achieved occupancy | 6.25% | **6.25%** |
| us/step | **3.05** | 7.49 |

The spill is gone and the kernel is **2.5x slower**. Occupancy does not move
either: 64 KB of shared per block simply replaces registers as the binding
resource.

So the old note was right and my layout theory was wrong. L1-cached local
memory really does beat a shared tile here — "global hits L1 with better
pipelining", as it said. Reverted.

**This is the second time today I doubted a measured rejection on a plausible
mechanism and lost.** The first was claiming chunkwise-WY had never been
measured when two A/B tables sat in this directory. A mechanism that explains
why someone else's result should have been different is a hypothesis; their
number is data.

## Where GDN actually stands

- The cost is **not** arithmetic (SM 11.57%), **not** DRAM (0.28%), and not
  removable by eliminating the spill.
- Occupancy is 6.25% and every cheap way to raise it just moves which resource
  binds: registers -> shared, or nothing at all.
- The one path left is reducing state PER THREAD — splitting the column across
  4 threads gives 32 floats each, which fits without spilling and without a
  large shared tile, at the cost of a 4-thread reduction in `k·S` and `q·S`.
  Barriers measured free here (two extra per token made it 8.5% *faster*), so
  the usual objection to that does not apply.
- Untried, and every hypothesis I formed today about this kernel has been
  refuted by measurement, so it is a candidate rather than a plan.

## The K split: three attempts, reverted — but its premise is now verified

Splitting the state column across KSP threads (KSP/thread rows each, a
KSP-way reduction through shared memory for `k·S` and `q·S`), behind a
`_GDN_KSPLIT` constant so KSP=1 is the shipped math.

KSP=1 came out wrong every time — out 125.9%, state 100.0% — through three
fixes: `T.serial(KSP-1)` replaced by a trace-time `range` (a zero-trip
`T.serial(0)` is not safe), and the `(2, KSP, V)` reduction buffer flattened to
1-D since every other shared buffer here is. Neither moved the number.
Reverted at the attempt limit.

**What the failures did establish, and it is the expensive half:**

- **The reduction path is free.** KSP=1 with the full shared-memory reduction
  and two extra barriers per token measures **2.99 us/step against the shipped
  3.05** — so the cost model that would kill this design does not hold. The
  barrier probe said the same independently: two dummy barriers per token made
  the kernel 8.5% FASTER.
- So the open question is purely a correctness bug in the reduction wiring, not
  whether the design can pay.

## Four tilelang traps, one shape

Everything that has silently produced wrong numbers in this kernel today is a
Python-level value leaking into traced control flow:

| written | what it did |
|---|---|
| `expr and vs == 0` | `bool()` on a symbolic operand is always true; collapsed to `vs == 0` |
| `T.serial(K // VB)` | symbolic bound, so `q_s[ki*VB + tv]` became a dynamic shared index and broke the layout |
| `T.serial(KSP - 1)` | zero-trip loop at KSP=1 |
| `T.alloc_shared((2, KSP, V))` | 3-D shared buffer among 1-D ones |

The rule that survives all four: **keep Python ints in Python** — `range()`,
`if` on a constant, a flat buffer — and let symbolic values appear only as data
indices. Where a refactor must be identity at the default setting, make it
identity by CONSTRUCTION (emit nothing) rather than by argument.
