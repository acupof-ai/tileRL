# The launch floor is ~10 us, not ~60, and the two are different quantities — 2026-09-06

> Status: Measured and shipped. The gate now tests utilization, the floor is a number, and
> all five citations carry it. Closes
> [a number with no instrument](2026-09-06-a-number-with-no-instrument.md).

## Context

Five files assert a "~60 µs eager launch floor regardless of shape" with no instrument
behind any of them; three cite one of the others
([a number with no instrument](2026-09-06-a-number-with-no-instrument.md)). It is the
stated reason task #21 says to measure in the captured graph first, so the figure decides
where kernel work gets aimed.

`scripts/probe_launch_floor.py` existed to settle it and had never run: it refused any card
with >1 GiB resident, and the V100 permanently holds the production endpoint at ~27 GB.

## The gate tested the wrong quantity

The refusal's own justification is **contention** — "a card serving traffic contends for
exactly that". The check was **residency**. Those come apart on this exact card: sampled
over 30 s the V100 reads **median 0%, max 2% utilization** with 27,306 MiB resident. The
endpoint is loaded and issuing nothing.

So the OPEN.md item asking for "an idle V100" was unsatisfiable as written, and the probe
had been unrunnable for that reason rather than for a real one. The gate now samples
`nvidia-smi` at 200 ms for 30 s (n≈150) and refuses on **median util > 5%**, reporting
residency as context. Residency alone is fine.

**The first version of that sampler was broken and I nearly reported its number.** It ran
in a Python thread and collected **n=2** over a 40 s run: the measurement loop holds the
GIL and each `nvidia-smi` invocation costs ~1 s. n=2 cannot distinguish an idle card from a
busy one, and its 34% median was as plausibly my own gemv as another process. Rewritten as a
detached `nvidia-smi -lms 200`, which raises n to 150. **Cost of the sampler on the quantity
being measured: 1.006x** (9.70 → 9.76 µs), measured rather than assumed.

## The floor

sm70, Tesla V100-SXM2-32GB, `/usr/bin/python3` torch 2.5.1+cu121, util n=150 median 0%:

| arm | µs/call |
|---|---:|
| `empty` (`torch.empty(1).zero_()`) | 11.28 |
| `tiny_add` | 10.13 |
| `inplace` | 7.58 |

The sweep flattens exactly where a floor predicts, and the knee is visible:

| elems | µs/call | GB/s |
|---:|---:|---:|
| 1 | 10.06 | 0.0 |
| 1 024 | 9.98 | 0.8 |
| 16 384 | 9.79 | 13.4 |
| 262 144 | 10.06 | 208.6 |
| 1 048 576 | 11.81 | 710.2 |
| 4 194 304 | 44.03 | 762.1 |
| 16 777 216 | 166.78 | 804.8 |

**Flat at ~10 µs across four decades**, then work-bound from 2^20 on. So "regardless of
shape" is right — the half nobody tested separately — and the magnitude is **6x** off.

## Why 60 was not simply invented

**`benchkit.timeit` reports amortized cost, and a per-call-sync harness reports something
larger.** `timeit` records CUDA events around a loop of N calls and divides by N, so the CPU
enqueues call k+1 while the GPU runs call k. Same kernels, three timer shapes:

| timer shape | tiny_add | gemv 1x5120x6144 |
|---|---:|---:|
| amortized (`timeit`) | 10.10 | 100.21 |
| per-call sync (wall) | 21.23 | 120.53 |
| isolated (events, 1×) | 25.60 | 119.81 |

Per-call is **2.1x** the amortized figure on a trivial kernel and **1.2x** on a real GEMV —
the fixed cost is the same ~11-15 µs in both, which is what a floor means. So there are two
defensible floors, ~10 and ~21-26 µs, and they answer different questions.

**Neither reaches 60.** No shape of this timer on this card produces it, so 60 is not a
sync-convention difference; it remains unsourced. The nearest measured relative on file is
~40 µs on H20 (`errors/2026-08-27-fp4-gemv-issue-bound-ncu.md`), and 60 is not that either.

## What it changes for #21

The premise survives with a smaller number. A microbench of a decode GEMV at ~100 µs carries
**~10 µs** of harness cost amortized, so **~10%** of a microbench reading is floor — not the
~60% that ~60 µs would imply. Small-N shapes are the ones where that fraction is largest,
which is why the instruction to measure in-graph first is still correct; it is just a
smaller correction than the number that motivated it.

## What the floor lets me settle, and what it does not

The floor makes one adjacent question answerable and shows a second is not.

**The remaining f32→f16 casts are launch-bound, so pro-rating their cost by count is
legitimate.** #21's live descendant is the 112 casts that
[elementwise writes the GEMV dtype](2026-09-02-elementwise-writes-the-gemv-dtype.md) did not
delete (`o_proj` 16, `out_proj` 48, `ab` 48 — counted from the checkpoint's `text_config`:
16 full-attention and 48 linear layers, both projections `[6144 → 5120]`). Pro-rating the
task's 1.64 ms/305 by count is only valid if a cast's cost does not track its width, and the
deleted set includes `down` at width 17408 while the remaining set tops out at 6144:

| width | M=1 | M=8 |
|---:|---:|---:|
| 5120 | 9.93 | 9.90 µs |
| 6144 | 9.87 | 9.73 |
| 17408 | 9.76 | 9.98 |

**17408/5120 is 0.983x at M=1 and 1.008x at M=8, against a 3.40x byte ratio.** Flat, so
width is irrelevant across every width a real cast uses and count is the right unit.

**The two bounds differ by 1.9x, and the gap is the graph — measured, not reasoned.**

| bound | value | per cast | what it is |
|---|---:|---:|---|
| count pro-rate | 0.60 ms/token | 5.4 µs | 112/305 × the task's 1.64 ms, from an **in-graph** profile |
| floor × count | 1.13 ms/token | 10.1 µs | 112 × the **eager** floor measured above |

I first wrote these up as an unresolved conflict, with the gap attributed to floor×count
"assuming every cast pays a full launch with none overlapping". **Both halves of that were
wrong.** `bk.timeit` is already back-to-back, so 10.1 µs/call *is* the overlapped rate; and
the real difference was sitting in this entry's own premise — the 1.64 ms came from a profile
of the **captured graph**, and a graph replay does not pay per-call launch cost. That is
exactly the thing #21's instruction is about.

So I measured it: the same 112 casts at the real widths, eager loop versus captured graph.

| arm | ms / 112 casts | µs per cast |
|---|---:|---:|
| eager loop | 1.093 | 9.76 |
| captured graph | **0.366** | **3.27** |
| what the graph saves | 0.727 | 6.49 |

**eager / graph = 2.99x.** The eager arm reproduces the floor (9.76 against 10.1), and the
graph arm is *below* the pro-rate's 5.4 µs. So the bounds were never in conflict: one was an
eager number and one an in-graph number, and the honest figure for the 112 casts in the
shipped path is **~0.37 ms/token**, with 0.60 ms an upper bound inherited from the older
profile's attribution.

**What is still open** is narrower than I claimed: not "which bound is right" but whether
the shipped graph's casts cost what an isolated graph of the same casts costs — the real
graph interleaves them with GEMVs that may hide some of it. That needs
`prof_decode_budget.py` by kernel, which needs the 27B resident: 5,500 MiB free against a
19 GB model, and evicting the production endpoint is not worth this number.

## Fix

- the gate samples utilization instead of residency, and refuses on median > 5%
- the probe prints all three timer shapes, so the next reader cannot pick the one that
  suits the argument without seeing the other two
- all five citations replaced with the measured figure and this entry

## Rule

**A floor quoted without its timer convention is two numbers wearing one label.**
Amortized-in-a-loop and per-call-with-sync differ 2.1x here on the same kernel, and both are
honestly called "the launch floor". Any fixed-overhead figure has to state which one it is.

Second: **a guard's threshold has to test the quantity its own justification names.** This
one said "contention" and measured "residency", which refused a valid card for six days and
made an OPEN.md item unsatisfiable. When writing a refusal, check that the condition and the
reason are the same condition.

Third, from reading the checkpoint: **an empty result with exit 0 is not an absence.** My
first two attempts to read the model dims printed nothing and returned 0, which reads as
"those keys are not in this config"; they are under `text_config`, and I was querying the
top level. A lookup that finds nothing has to distinguish "not there" from "not where I
looked" — print the keys you did find.

Fourth, and the one this entry earned the hard way: **two numbers that disagree are not
automatically an open question.** I published 0.60 and 1.13 ms as unresolved bounds and gave
a mechanism for the gap that was wrong in both halves — while the resolving fact was already
stated in this same entry, three sections up, as the reason #21's instruction exists. Before
calling a discrepancy unresolved, check whether the two numbers were measured under
conditions this document already distinguishes. **A ratio between two of your own numbers is
a claim about them, and it needs the probe that separates them, not a plausible story.**

## Results

No runtime change; instrument and documentation only.

| date | commit | machine | quantity | value |
|---|---|---|---|---|
| 2026-09-06 | f0e58b3 | V100 sm70, util median 0% | floor, amortized | **10.1 µs** |
| 2026-09-06 | f0e58b3 | V100 sm70, util median 0% | floor, per-call sync | **21.2 µs** |
| 2026-09-06 | f0e58b3 | V100 sm70, util median 0% | floor, isolated events | **25.6 µs** |
| 2026-09-06 | f0e58b3 | V100 sm70, util median 0% | f32→f16 cast, 5120 / 6144 / 17408 wide | **9.93 / 9.87 / 9.76 µs** at M=1 |
| 2026-09-06 | f0e58b3 | V100 sm70, util median 0% | 112 casts, eager vs captured graph | **9.76 vs 3.27 µs/cast, 2.99x** |
| — | — | — | the retired assertion | ~60 µs, unsourced |
| — | — | — | the 112 casts in the shipped path | **~0.37 ms/token** in-graph; 0.60 ms is the older profile's upper bound |

Raw artifacts: `scripts/probe_launch_floor.py` (floor and timer shapes),
`scripts/probe_cast_width.py` (the width sweep, which prints the byte ratio beside the time
ratio so a flat result cannot be mistaken for a bandwidth reading),
`scripts/probe_graph_vs_eager.py` (the graph arm, which is what turned two "conflicting"
bounds into one number and one convention).
