# The 112 casts cost 0.349 ms/token, serialized by stream order rather than occupancy — sm70, 2026-09-06

**Date:** 2026-09-06
**Arch:** sm70 (Tesla V100-SXM2-32GB), beside ckl's resident 27B endpoint (27,750 MiB, 0% util)
**Commit:** 0489777 (#172) for the pair/sweep arms, 363de2a (#175) for the stream-order arm — the merged ancestors of the branches they ran on; the probes themselves land with this entry
**Verdict:** **reject** — task #21's last live descendant closes with a number instead of a bound.

## Context

`backend.py:696` casts X to f16 inside the fp4 GEMV chunk loop, once per launch. #21's
descendant, the one OPEN.md row that survived
[the grep that retired #21's three figures](2026-09-05-task-21-numbers-do-not-survive-a-grep.md),
asked what the remaining 112 of those cost in the shipped graph. The launch-floor entry
measured them in an **isolated** graph at 3.27 µs each, 0.366 ms/token, and recorded it as an
**upper bound** with the reason stated: the shipped graph interleaves each cast with the GEMV
that consumes it, and a DRAM-bound GEMV was assumed to have idle SMs a cast might occupy. That
assumption is tested and refuted below — the binding constraint is stream order, not occupancy.

That row said "blocked on capacity, not a job" — 19 GB of model against ~4.4 GB free. True of
`prof_decode_budget.py`, which loads the model. **Not true of the question:** a cast and its
GEMV are two kernels, and a 6144×5120 f16 weight is 60 MB.

## What was measured

`scripts/probe_cast_hides_in_gemv.py`, three captured graphs at the real widths (o_proj 16 and
out_proj 48 at 6144→5120, ab 48 at 5120→96), M=1, weights from a 244 MiB pool cycled across
the 112 launches:

| arm | ms | µs each |
|---|---:|---:|
| 112 casts alone | 0.366 | 3.27 |
| 112 GEMVs alone (X pre-cast) | 9.393 | 83.87 |
| 112 (cast → GEMV) pairs | 9.739 | 86.96 |

So the cast **adds 0.346 ms** to a GEMV sequence that costs 9.393 — 3.09 µs each against
3.27 µs measured alone. **The cast does not hide.**

## The 6% that looked hidden is not a measurement

3.09 against 3.27 is a 5.6% difference, and it was tempting to report "6% hides". Then the
first arm was re-run **last**, same process, same 112 casts:

| reading | ms | µs each |
|---|---:|---:|
| cast only, first position | 0.366 | 3.27 |
| cast only, last position | 0.309 | 2.76 |
| **same arm, spread** | | **18.4%** |

**The same-arm spread is 3.3x the difference between arms**, so the 6% is inside the noise of
the instrument and says nothing. Reported as unresolvable rather than as a small effect.

## The figure that survives is the slope

A count sweep separates the per-cast rate from whatever the fixed cost of a replay is:

| casts | ms | µs each |
|---:|---:|---:|
| 0 | 0.013 | — |
| 28 | 0.081 | 2.89 |
| 56 | 0.157 | 2.80 |
| 112 | 0.309 | 2.76 |
| 224 | 0.623 | 2.78 |
| 448 | 1.321 | 2.95 |

**Marginal cost 3.12 µs per cast** (224→448), and an empty graph replays in **13.1 µs**. The
intercept is real but small — 0.12 µs per cast spread over 112 — so it is not what moved the
112-cast arm between positions; the position spread is.

This sweep is also what caught the arm difference in the first place. The first version of the
probe doubled the cast count once, expecting ~2.00x if same-stream kernels simply serialize,
and got **1.70x**. That is what a fixed intercept plus a noisy 112 reading looks like, and it
is why the sweep replaced the single doubling.

**112 × 3.12 µs = 0.349 ms/token**, against the 0.366 previously published as an upper bound.
The bound was 5% high, and it was high for the reason bounds usually are — it included a share
of the replay cost — not because the cast was hiding.

## The reason it does not hide is stream order, not occupancy

The paragraph above says a DRAM-bound GEMV has idle SMs a cast might occupy, and the additive
result was first written up as evidence the GEMV had no room. **That attribution was untested
and it is wrong.** Kernels issued into one stream execute in issue order, so a cast placed
after a GEMV cannot overlap it at any occupancy — a cheaper explanation the measurement did
not distinguish.

`scripts/probe_cast_stream_order.py` discriminates by putting the casts on a **second stream**,
concurrent with the GEMVs, inside one captured graph (a graph records cross-stream parallelism).
The GEMVs consume the pre-cast `xs16`, so no data dependency forces the order — the question is
whether the hardware *can* run them together:

| arm | ms | cast adds |
|---|---:|---:|
| 112 GEMVs alone | 9.379 | — |
| serial: cast → GEMV, one stream (the shipped shape) | 9.718 | **+0.339** |
| parallel: casts on a side stream | 9.496 | **+0.116** |

**66% of the serial add disappears when the cast is allowed to run concurrently.** So the GPU
does have room; what serialized the cast was the stream, and "the GEMV is too occupied" is a
claim this repository should not carry.

The control that makes those numbers readable: the GEMV arm re-run last reads **9.380 against
9.379, a 0.0% same-arm spread**, and the effect under test (serial − parallel = 0.223 ms) is
**993x** that noise. The noisy arm in the earlier probe was the 0.366 ms *cast* arm, not this
one, so the spread had to be re-measured on the arm actually used as the baseline here rather
than carried over.

**The verdict does not move.** The shipped decode graph is one stream, so the shipped path pays
the full 0.339 ms — 0.349 ms/token at the marginal rate, 1.33% at ctx 8192. What changes is what
may be claimed about *why*, and one branch this opens: a side-stream cast is a real 0.22 ms/token
lever that needs no dtype change. Not pursued, for the same ratio — 0.85% of a token for a
second stream inside a captured decode graph, which is a correctness surface (capture order,
event dependencies) far larger than the win.

## Verdict: reject, with the cost priced

Against the served token at three contexts
([decode reaches 32K](../wins/2026-09-04-decode-reaches-32k-and-the-tick-slope-grows.md)):

| ctx | ms/token | casts | share |
|---:|---:|---:|---:|
| 8192 | 26.3 | 0.349 | **1.33%** |
| 16384 | 37.5 | 0.349 | 0.93% |
| 32768 | 65.3 | 0.349 | 0.54% |

**1.33% at the most favourable context is not worth a kernel change.** And reading the call
site closes it harder than the arithmetic does: `backend.py:663-666` says the cast **is** the
optimisation. Handing the kernel X pre-packed as f16 replaces the kernel re-reading X per block
and converting inside the tile loop — measured **4.2x at M=32 and 1.1-1.45x at M=1, bit-exact,
both paths round to nearest f16**. The in-kernel cvt is the arm this cast already beat.

So the framing in the OPEN.md row — "attribute the 112 casts, they may be removable" — had the
sign wrong. There is no version of this path with no conversion; there is a conversion at the
call site costing 3.12 µs, and a conversion inside the tile loop that costs 1.1-1.45x of the
whole GEMV. `backend.py:298-312` states the same thing from the dtype side: sm70's `io` is f32
by invariant, and "sm70's fp16 GEMV does its f32->fp16 cvt inside the kernel, not via io".

What remains theoretically available is folding the cast into the GEMV kernel's **prologue** —
one launch instead of two, keeping the pre-packed layout. That is worth at most the 0.349 ms,
1.33% of a token, for a change to the arch-gated dtype contract that
`wins/2026-09-02-kv-pool-dtype-is-the-kernel-abi.md` exists to protect. Rejected on that ratio.
The 5.9% / 1.64 ms figure that made this look worth chasing was **305 casts**, before the count
fell to 112.

What this does NOT reject: the cast count itself. If a later change puts the count back up
toward 305, 3.12 µs × 305 = 0.95 ms/token = 3.6% at ctx 8192, which is a different decision.
The slope is the reusable number.

## Limitations, stated

- **Dense f16 matmul, not the fp4 kernel.** The fp4 GEMV needs quantized weights from the
  checkpoint, which is the capacity problem. Occupancy turned out not to be the binding
  variable — stream order is, and the shipped decode graph is one stream regardless of which
  kernel runs in it — so the substitution does not affect the serial figure the reject rests
  on. It does bound the side-stream branch: 66% recovery is the dense GEMV's, and the fp4
  kernel could leave more or less room.
- **244 MiB weight pool, 4 per shape, cycled** — 112 distinct weights would be 4.07 GB against
  4.4 GB free beside the endpoint. Asserted larger than 4× L2 so a weight cannot be resident at
  its next launch, but it does not reproduce the shipped path's total footprint.

## Rule

**A bound quoted from an isolated arm carries that arm's fixed cost.** 0.366 was 5% above the
marginal 0.349 for exactly that reason, and the fix is a count sweep, not a better single
measurement.

**Before reporting a difference between two arms, re-run one arm in the other's position.**
The 5.6% pair-vs-alone difference sat inside an 18.4% same-arm spread. Nothing about the
ordering was suspicious; the control is cheap and it is the only thing that sizes the claim.

**"Blocked on capacity" is a property of an instrument, not of a question.** This row sat open
because the *profiler* needs the model. The question needed 244 MiB.

**Read the call site before pricing its removal.** I measured the cast three ways and wrote the
reject on the arithmetic, then read `backend.py:663` and found the cast is a **deliberate
optimisation with its own measured win** (4.2x at M=32, 1.1-1.45x at M=1) over the in-kernel
cvt. The measurement was right and the framing it was answering was wrong: there is no
no-conversion arm. Ten lines of comment at the call site said so, and the same fact appears a
second time at `backend.py:301`. The order should have been read-then-measure.

**An additive result names no mechanism.** "The cast adds its full cost, therefore the GEMV has
no idle room" was written from one measurement that is equally consistent with the cheaper
explanation — one stream executes in issue order. Putting the casts on a second stream recovered
66%, so the GPU had room all along. A serialization has at least two possible causes, the
software ordering and the hardware saturation, and an experiment that varies neither cannot
choose between them. **Before attributing an additive cost to occupancy, remove the ordering
constraint and see whether the cost survives.**

## Results

| date | commit | host | target | model | shape | metric | value |
|---|---|---|---|---|---|---|---|
| 2026-09-06 | 0489777 | V100 32GB | cuda sm70 | f16 dense, real widths | 112 casts, M=1 | marginal µs/cast | **3.12** |
| 2026-09-06 | 0489777 | V100 32GB | cuda sm70 | f16 dense, real widths | 112 casts, M=1 | ms/token | **0.349** |
| 2026-09-06 | 0489777 | V100 32GB | cuda sm70 | f16 dense, real widths | empty graph | replay µs | 13.1 |
| 2026-09-06 | 0489777 | V100 32GB | cuda sm70 | f16 dense, real widths | 112 pairs | fraction hidden | **unresolvable** (5.6% vs 18.4% spread) |
| 2026-09-06 | 363de2a | V100 32GB | cuda sm70 | f16 dense, real widths | 112 casts + 112 GEMVs | cast adds, one stream | **+0.339 ms** |
| 2026-09-06 | 363de2a | V100 32GB | cuda sm70 | f16 dense, real widths | 112 casts + 112 GEMVs | cast adds, side stream | **+0.116 ms** |
| 2026-09-06 | 363de2a | V100 32GB | cuda sm70 | f16 dense, real widths | 112 casts + 112 GEMVs | recovered by concurrency | **66%** |
| 2026-09-06 | 363de2a | V100 32GB | cuda sm70 | f16 dense, real widths | GEMV arm, re-run last | same-arm spread | 0.0% (effect 993x noise) |

No runtime change — two probes, this entry, and one OPEN.md row removed.
