# B=16 is 23.8% cheaper per token and fits with 57 GiB spare, and the first clock said the opposite — 2026-09-08

**Status:** both questions **settled**. B=16 fits (peak 30.13 GiB of 95.2) and costs **7.930
ms/token against B=8's 10.409, 23.8% cheaper**, measured with zero JIT inside the timed steps.
The first attempt reported the opposite sign because 47% of one arm's wall clock was TileLang
compiling, and the memory bracket I gave a peer before the run did not contain the answer.

## Context

`tilerl-0a`'s static dispatch table says wgmma's M granularity in this repo is 16, not 64, so
B=8 fills the tensor core 50% and B=16 fills it 100% with no valley in between — the sweet
spot is exactly 16, since past it a request enters the prefill bucket and bM jumps to 32
(53% fill). The bandwidth side says the move is nearly free: from the
[per-tick floor table](2026-09-08-four-rulers-the-wrong-size-for-their-object.md), only the
per-batch terms double, **+5.6% of bytes for 2x the tokens**.

Capacity is the one term that table does not compute, and the training path never measures it
— `cli.py` passes `num_blocks` explicitly, which is what skips `_fit_blocks`, the only code
that reads free memory (`engine.py`, and 0a documented the omission in #309).

## The measurement

Card 2, H20, 95.2 GiB total. 27B NVFP4, all 64 layers, gen 1024, prompt 256, `decode_graph`
off and `NoPrefixStore` — the shape `_require_on_policy` demands. Pool sized by
`_train_adapters`' own formula, not hand-picked.

```
#   B   pool   used  peak GiB  resv GiB  free GiB  fin   len    secs
    8    680      0     25.99     27.82     67.40    8  1024   217.5
   16   1352      0     30.13     37.98     57.24   16  1024   507.9
```

**B=16 fits: peak 30.13 GiB, reserved 37.98, 57.24 GiB left, all 16 rollouts at the full
1024.** Both arms ran 1030/1038 ticks for a 1024-token generation, so neither queued a
second wave — the slots held the whole group. The recommendation is executable.

## 1. The bracket I gave a peer was not a bracket

Before the run I derived two bounds from the B=8 arm and sent them as "the answer will land
between these": **26.65 GiB** if only the KV pool doubles, **28.76 GiB** if the entire
non-weight term doubles — described as the worst case.

Measured **30.13**, outside both. The non-weight term went 2.77 → 6.91 GiB, **2.49x for 2x
B**. It is superlinear, so "everything doubles" was never an upper bound.

The probe's own `base`/`after_build`/`peak` triple locates where I was wrong, and it is not
where I looked:

| | B=8 | B=16 | delta |
|---|---:|---:|---:|
| base (weights) | 22.91 | 23.07 | +0.16 |
| pools (`after_build - base`) | 1.87 | 3.74 | **+1.87** |
| transient (`peak - after_build`) | 1.21 | 3.32 | **+2.11** |

**The pool derivation was right to 1.05x** — I predicted +1.78 GiB (GDN state 1.12 from
9→17 slots × 48 layers × 48 value heads × 128 × 128 f32, KV 0.66 from 680→1352 blocks ×
512 KiB bf16) against +1.87 measured. **The entire miss is transient: 1.21 → 3.32 GiB,
2.74x.**

**What the transient is, I have not established.** Ruled out by arithmetic, each too small by
three orders: conv state (+0.044 GiB), logits with the sort's buffers (+0.023), attention
partials (+0.00003), and prefill activations — `max_num_batched_tokens` is 512, so a prefill
tick carries 512 tokens at either B and its MLP activation is 0.033 GiB both times. The
per-token decode activation is 0.31 MiB at B=16. Nothing I can name accounts for GiB.

Allocator behaviour is the remaining candidate and the numbers are consistent with it —
`reserved - peak` is 1.83 GiB at B=8 and 7.85 at B=16, **4.29x**, growing faster than the
allocated peak's 1.16x. But consistent-with is not measured, and the instrument for it is
`torch.cuda.memory_stats()`'s size bins, which this probe does not read.

So the term is recorded as unlocated. **A decomposition that accounts for half of a delta
looks exactly like a correct one** — the same failure as the 6.7 GB gap in my weight floor,
and the general form of `tilerl-48`'s lesson that three successive per-op tables each summed
to the measured step while the attribution was wrong.

**The capacity verdict does not rest on the decomposition.** It rests on 30.13 measured
against 95.2, with 57.24 GiB spare. Even if the unlocated 2.11 GiB doubled again at B=32,
the answer would not change.

## 2. The wall clock says B=16 is slower, and that number is unusable

**Re-measured with the gate, and it reverses: B=16 is a net win.** Same card, same shapes,
`prof_grpo_step.py --steps 5`, `warm_compiles: 0` in both arms:

| | B=8 | B=16 | ratio |
|---|---:|---:|---:|
| sec/step | 85.27 | 129.93 | **1.524x** (net win under 2.000) |
| **ms/token** | **10.409** | **7.930** | **0.762x — 23.8% cheaper** |
| rollout/token | 7.690 | 5.226 | 0.680x |
| decode/token | 7.263 | 4.486 | **0.618x** |
| train/token | 2.719 | 2.704 | 0.994x |

The contaminated figures were 26.55 and 31.00 ms/token — **2.55x the clean B=8 value, and the
wrong sign on the comparison.** A JIT-dominated clock did not merely add noise; it inverted
the verdict.

The decode row is what `tilerl-0a`'s dispatch table predicts: fill goes 50% to 100% at
wgmma's M granularity of 16, so the ideal is 0.500x per token and **0.618x realizes 76% of
it**. The other rows check that table's silences: `train/token` is 0.994x because `--micro 1`
runs one row per backward and cannot benefit, exactly as it should be; `mixed` grew 3.692x
because 16 rows admit across more ticks, so more ticks carry both phases.

**This does not contradict 48's finding that the per-call rate falls past M=8** (fp4 832.8 to
531.7 GB/s at M=16). Both hold and they compose: batch amortization beat the per-call rate
drop. A per-call rate and a per-token cost are different quantities, and the second is what a
training step pays.

**What was not read: `clocks.sm` inside the timed window.** 48 raised this for their own
microbenchmarks — an idle card sits at 345 MHz against a 1980 MHz maximum, 5.7x — and then
measured it away (identical 1980 MHz and 0.081 ms from 10 to 20000 warm-ups, the card at full
clock as soon as the 27B is resident). It cannot be a first-order term here either: each arm
runs 184–232 s of step 0 before the timed steps and 85–130 s per step, five orders past a
sub-second ramp. The honest limit is that a *thermal* excursion over a 130 s step remains
unmeasured — and it would hit the longer arm harder, understating B=16, so **0.762x is
conservative under that failure mode** rather than flattered by it.

### The contaminated arm, kept on record

B=8 is 26.55 ms/token, B=16 is **31.00 ms/token — 1.168x worse**, the opposite of the
dispatch prediction. Reporting that as a refutation would have been wrong.

Counted from the log's timestamps: **183 TileLang compiles, 379 s total.**

| arm | compiles | compile s | reported secs |
|---|---:|---:|---:|
| B=8 | 42 | 142 | 217.5 |
| B=16 | 141 | 237 | 507.9 |

**B=16 compiled 3.4x as many kernels** — each new decode width recompiles
`paged_attention_decode`, `gdn_decode_fused` and `write_tokens`, and the B=16 arm reaches
more widths. 237 s of 507.9 is 47%.

Subtracting compile time does not rescue it either: compilation and execution interleave in
one wall clock and this probe has no separate timer for them, so 271 s vs 76 s is not a
throughput comparison. **A wall clock dominated by JIT can neither confirm nor refute a
prediction about kernel efficiency.** Whoever prices B=16's throughput needs a warm arm with
the compile cache hit — a different measurement.

## 3. The probe's own `pool_used` column is empty

Both arms report 0. I read `engine.stats()` after the drain, and a request releases its
blocks on finish, so the column carries no information. Fixed by sampling mid-drain every 64
ticks and keeping the maximum — every tick would take the step lock and time the probe
instead of the engine. The peak columns are unaffected: `max_memory_allocated` is a high
water mark.

## The coupling defect this measurement exposed

`cli.py` built the training engine with `num_slots=8, max_batch=8` and a pool sized `*8` —
three literals — while `--group` is a settable flag defaulting to 8. So **`--group 16` did
not produce B=16.** `grpo_loop` submits the whole group before draining, and an engine with
fewer slots neither raises nor drops: `Engine.__init__` documents that the excess queues into
later ticks, two waves of 8 at half the rows per tick.

That is silently the 50% tensor-core fill the whole B=16 recommendation exists to remove.
**The harm is not the slowdown — it is that an experiment asking "does a wider group help"
would have answered "it does not", with nothing visible to explain why.**

Fixed by sizing all three from `max(args.group, 1)`, and guarded permanently in
`train._require_group_fits`, beside `_require_on_policy`. Three reasons for those choices:

- **In `grpo_loop`, not the CLI.** `grpo_loop` is what submits the group, and three scripts
  (`probe_pad_histogram.py`, `recapture_arms.py`, `step_phase_split.py`) build their own
  engines — a CLI-side check misses exactly the callers most likely to run this experiment.
- **Reads `usable_slots`, not `max_batch`.** The slot is what a request holds from submit to
  finish; `Engine.__init__`'s own comment says `max_batch` is not the real ceiling.
- **Raises rather than warns.** A defect whose symptom is a wrong conclusion needs an
  assertion, not a warning that a caller may not read. A defect that can fabricate a
  refutation is worth a permanent guard, not one fix.

## The guard narrowed 12 older callers, and one of them was frozen

A new precondition turns every test past it into a test of the precondition, so the 12
`grpo_loop` call sites were run rather than reasoned about. One failed and one was silently
weakened.

**The failure:** `test_grpo_length_buckets_preserve_real_token_loss_and_gradients` drives
`grpo_loop` with a `SimpleNamespace` stub, which has no `usable_slots` —
`AttributeError`. Fixed in the stub, not with a `getattr(engine, "usable_slots", ...)`
default in the guard: a default makes the guard pass for every caller, which is the
capability-check trap `_require_on_policy`'s own docstring names.

**The weakening:** `test_ledger.py`'s engine-config test asserted `slots == 8` — a literal
that was correct only because the CLI's was too. After the fix production computes `slots`
from `--group` and the test would have kept passing against a frozen 8 while measuring
nothing. Now asserted as `slots == 2 == max_batch` under `--group 2`, so it moves with the
flag.

## Rules

- **A "worst case" derived by scaling every term linearly is not an upper bound.** Something
  in the system was superlinear; the bound has to come from a measurement or from a term-by-
  term argument that the scaling is at most linear.
- **Split a memory delta into pools and transient before attributing it.** `base`,
  `after_build` and `peak` cost three lines and turn "my prediction was 2.36 GiB off" into
  "my pool math was right to 1.05x and the transient is unlocated".
- **A wall clock that contains a JIT is not a throughput measurement, and it can invert a
  verdict rather than blur it.** The contaminated arms said B=16 was 1.168x worse per token;
  clean they say 0.762x — the wrong sign, not a wide error bar. Count the compiles inside the
  timed steps and refuse to report when the count is nonzero; subtracting afterwards does not
  work because compilation and execution interleave in one clock.
- **Compare per-token cost, never per-step.** sec/step rises with the group whatever the
  efficiency, so two widths compared on it only show that the wider one did more work.
- **Sample a pool's occupancy while it is occupied.** A post-drain `stats()` reads 0 and that
  0 looks like a measurement.
- **A guard belongs where the resource is consumed, not where the flag is parsed.** The CLI
  is one caller.
- **Satisfy a new precondition in the stub, never with a default inside the guard.** A
  `getattr` fallback makes the guard pass for everyone, which is the same as not having it.
- **A literal in a test is only safe while production also has a literal.** When the value
  starts being computed, the assertion has to be computed too or it goes green over nothing.
