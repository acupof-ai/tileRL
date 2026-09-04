# The depth-4 stalls are TileLang compiles, and my own "not a shape set" entry was wrong

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 + 456M draft, B=1, ctx=1024, wikitext
**Run:** `t74` — `--prompts 6 --batch 1 --tokens 128 --time-draft`, 6 disjoint passage
groups × 4 depths, at `1f58293`
**Tasks:** #22 (closes REJECTED), #72 (the B=1 arm)

## Why 6 groups instead of 3

Every previous B=1 row on this arch came from `--prompts 3`, which is groups p0/p1/p2.
The sm90 peer's boundary claim — "nothing before p3 ever blows" — is invisible to a
3-group run by construction. My earlier "all groups within 1.075x, so B=1 is immune" was
not evidence of immunity; it was a run that never reached p3. This is the first B=1 sweep
that does.

## The four depths, per group

Depths 1-3 are flat. Depth 4 is not, and the excursions are at p3 and p5:

| depth | tick spread over 6 groups | draft spread | `tick − draft` spread |
|---|---:|---:|---:|
| 1 | 1.0879x (p1 alone) | 1.5538x | **1.0020x** |
| 2 | 1.0107x | 1.0141x | 1.0127x |
| 3 | 1.0123x | 1.0218x | — |
| 4 | **4.12x** (p3), 2.08x (p5) | 16.1x | 3.3989x |

Depth 4 in full:

```
 p     tick   draft  tick-draft  tok/fwd   retry   resv
 0    98.79    5.61       93.18     2.59       0  19.98G
 1   100.79    6.38       94.41     2.59       0  19.98G
 2    98.61    5.82       92.79     2.15       0  19.98G
 3   405.55   90.17      315.38     3.34       0  19.98G
 4    98.43    5.60       92.83     2.95       0  19.98G
 5   204.55   37.86      166.69     2.59       0  19.98G
```

The four clean groups sit within 1.0240x of each other. Both blown groups are at p≥3, so
**the peer's boundary reproduces on sm70 at depth 4** — while being inverted at depth 1,
where the only excursion is p1 and p3/p4/p5 are the flattest three in the set. The
boundary is therefore not a property of position; it is a property of whatever the run
does at that depth.

## Both proposed mechanisms are dead, and the second one was mine

**The prefill rectangle cannot be it.** `measure()` opens its timed window only after the
drain loop confirms every row is `_PHASE_DECODE` (`ab_draft_depth.py:195-199`) and
submits nothing inside it, so `engine.py:776 if decodes and prefills` cannot fire — and
`:240-241` already raises `SystemExit` on any nonzero `mixed_forwards` delta, which every
printed row above passed. Capping `max_num_batched_tokens` would cap a term that is
already absent: both readings of that experiment predict no change, so it is null rather
than weak, and it was not run.

**Allocator fragmentation is refuted by the discriminator registered before the run.** I
predicted: fragmentation ⇒ blown groups carry `num_alloc_retries > 0` and reserved bytes
climb across p0→p5; retries == 0 with flat reserved refutes it. Measured: **retries 0 in
all 24 group-windows**, and reserved is **19.98G in all six depth-4 groups**, flat to the
digit through a 4.12x stall.

## What it is: 6 TileLang compiles, all inside the depth-4 window

This invocation has the INFO logger the ctx=2048 run lacked, so the count is real:

```
13:02:00  write_tokens_f32          13:02:08  write_tokens_f32
13:02:02  paged_attention_split     13:02:10  paged_attention_split
13:02:45  write_tokens_f32          13:02:47  paged_attention_split
```

Three matched pairs, 1-3 s apiece, **12 s of compile wall time**. The depth-3 row printed
at log line 58; every compile line is at 62+, i.e. after depth 3 finished and inside
depth 4.

Depth 4 is the only depth that reaches **rung 8** (`r8x244` of 287 ticks; depths 1-3 live
on r2/r4). A new rung means new `(B, S)` shapes for the two kernels that take
`seq_q_lens` — which are exactly the two that compile.

The excess accounts for the compile time from both sides:

| | p3 | p5 | total |
|---|---:|---:|---:|
| draft excess `(draft − 5.60) × 138 fwd/group` | 11.65 s | 4.44 s | **16.09 s** |
| tick excess `(tick − 98.43) × 47.8 ticks/group` | 14.69 s | 5.08 s | **19.77 s** |

against 12 s measured. The two routes agree to 18.6% and both land within 1.6x of the
logged compile time. Attributed, not proven: no run has yet been done with the cache
pre-warmed for rung 8 and the stall absent.

## The two-instrument argument, and why it is not the circularity I warned about

At depth 1 p1: tick +3.06 ms, draft +3.10 ms, agreeing to 1.3%, with `tick − draft` at
29.29-29.35 across all six groups — a **0.20% spread** through a 1.0879x tick excursion.
At depth 4 the same shape holds 15x larger: `tick − draft` moves only because the draft
term is subtracted from a tick that grew by the same amount.

This is a genuine attribution, unlike the derived-verify figure I cautioned the peer
about. There, verify was *defined* as tick minus its own rung's draft, so it could not
fail to absorb an excursion — which is why verify read 48.30/48.32/47.92 across ticks
that moved 11x. Here the draft is timed by CUDA events and the tick by wall clock: two
independent instruments, and the whole excursion lands in one of them.

## Verdict for #22: block-parallel drafting REJECTED, on the clean pair this time

Depths 2 and 3 both dispatch **rung 4** (`d2: r2x4 r4x333`, `d3: r2x3 r4x299` — 98%+),
so verify is held constant and the only difference between them is one draft forward.
That is the subtraction this script exists to make, and it is the first time it has been
made on 6 groups with the draft timed directly.

```
draft forward, differenced   61.25 − 56.42 = 4.83 ms
draft forward, CUDA events   5.54 ms (depth 3), 5.72 ms (depth 2)
```

The two instruments are 12.8% apart. The differenced figure is the noisier by
construction (operand/difference amplification, measured 12.9x in this file's history),
so the timed one is used below — and the verdict does not depend on the choice.

At the shipped depth 3, where a block-parallel head removes 2 of 2.93 draft forwards:

| | |
|---|---:|
| draft share of the tick | **26.5%** |
| tick ceiling if the draft were free | 1.2115x |
| collapsed tick | 50.56 ms |
| tok/s at depth 3's measured 2.55 tok/fwd | **50.4** |
| depth 1, measured | **50.1** |
| a **free** parallel head vs just setting depth 1 | **1.0067x** |

1.0067x is **inside this script's own 1.16% noise floor** — the threshold it prints
"inside the noise floor" against. So a perfect, zero-cost block-parallel head is
indistinguishable from `--spec-depth 1`, which is a config flip.

At depth 2 a free head loses outright: verify r4 45.43 + one draft forward 5.72 = 51.15
ms at 2.27 tok/fwd = 44.4 tok/s, **0.8858x** of depth 1.

Break-even, against depth 1's 50.1 tok/s:

| depth | collapsed tick | needs tok/fwd | has | |
|---|---:|---:|---:|---|
| 2 | 50.94 ms | 2.552 | 2.27 | FAIL |
| 3 | 50.56 ms | 2.533 | 2.55 | 1.007x |

**The reason is the rung, not the draft.** Verify r2 (depth 1) is 29.40 ms and verify r4
(depth 2) is 45.41-45.43 ms: a **16.03 ms rung step**, which is 2.89 draft forwards'
worth and 1.50x everything a parallel head could remove at depth 3. A parallel head
removes draft forwards; it cannot remove the rung step.

And the head is not free. A head emitting W tokens in one forward reads a W-row output
instead of one row, which on this arch is the rung ladder again — the draft's own GEMV
goes from M=1 to M=W. The trunk's measured r2→r4 step is 16.03 ms; a smaller model pays a
smaller version of the same shape, not zero.

Same direction as #30 (tick 88% GPU-bound) and as the earlier 3-group verdict, but this
one rests on the clean rung pair with the draft timed, at 6 groups, carrying every group.

## The depth default: depth 1 wins by 1.204x, and p1 is kept

The script's own summary line: `B=1: best depth 1 at 50.1 tok/s, shipped 3 at 41.6
(1.204x)`.

Depth 1's pooled 35.56 ms carries p1's 38.10 ms excursion, which costs the mean 1.4%
(35.05 → 35.56 ms, 50.9 → 50.1 tok/s). It is kept. Dropping it would be selection on the
dependent variable — the error the sm90 peer correctly identified in their own 1.078x,
where groups were dropped by the very quantity being compared.

## The corpus spread is wider than this file documents

The docstring records "acceptance varies 15.8% between wikitext passages (2.15 to 2.49
tok/fwd at W=4)". Six groups read **2.05-2.59 at depth 2 (26.3%)** and 2.15-3.02 at depth
3 (40.5%). A 3-group run samples 3 of 6 passages, so 15.8% is a floor on the spread, not
the spread. Any depth verdict resting on a single group is resting inside that band.

The tick/acceptance discriminator separates the two failure modes cleanly here, and both
appear in one run: depth 1 p1 moved the **tick** 1.0879x with acceptance flat (1.76 in a
1.69-1.84 band) — a compile; depth 2 moved **acceptance** 1.2634x with the tick flat at
1.0107x — corpus.

## Correction to `errors/2026-09-04-the-recompiles-reproduce-and-are-not-a-shape-set.md`

That entry's central claim is that the compile rate is **steady** — "~1 pair per 12 s,
flat across every 30 s bucket for the whole span" — and concludes "a finite shape set
exhausts; this does not", declaring the shape hypothesis dead and refusing to name a
cause.

This run contradicts it on sm70. **6 compiles total, 3 pairs, all within 50 s at one
depth transition, then zero across the remaining ~9 minutes and 287 further ticks.** That
is exhaustion, and it is the shape hypothesis behaving exactly as predicted: depth 4
introduces rung 8, the two `seq_q_lens` kernels compile for it, and the set is then
closed.

So the 269-compile B=8 run is the outlier that needs explaining, not this one. What
distinguishes them is not yet measured; the honest statement is that "the rate is steady"
was true of that one run and was generalized into a mechanism claim it could not support.
The B=8 arm cannot run on this card at any depth (`num_slots = max(batches)+1` hardcoded
at `:267`, `step_states` 0.141 GiB/slot/step on this geometry), so the comparison needs
the sm90 arm or a harness that pre-visits shapes.

## Rule

A stall that vanishes from `tick − draft` while the draft term carries all of it has been
*attributed*, not merely observed — but only when the two terms come from independent
instruments. The same subtraction against a derived term proves nothing, because a
derived verify absorbs nothing by construction.

And "the rate is steady" is a claim about a distribution, so it needs more than one run
before it kills a hypothesis. Mine killed the shape hypothesis off a single B=8 log, and
a 6-group B=1 run brought it back 40 minutes later.
