# Four ways a measurement's apparatus was the wrong size for its object — 2026-09-08

**Status:** the sampling optimization is **rejected**; the mechanism it was built on does not
hold on the card. What survives is method, and one open gap in my own derivation.

## What happened

`tilerl-48` attributed 27.2 s of a 30.4 s rollout to `_sample_commit` — about two thirds of a
GRPO step. I was asked to find the mechanism from the code while they measured on the card.

I found one: `reference.py:1207`'s `top_p_probs` calls `torch.sort` over the **whole
vocabulary** (248320) when the nucleus that gets sampled is 43 tokens. Measured on CPU at
B=8, `torch.sort` was 20.53 ms and the per-row `multinomial` 18.69 ms, against 0.52 ms for
`log_softmax` and 0.70 ms for `topk(k=64)`. That accounted for the 26.6 ms/tick almost
exactly.

On the H20 it accounts for **4.0%**. 48 measured `sort` at 0.274 ms and the per-row loop at
0.790 ms — 1.065 ms of the 26.6 ms. The optimization's ceiling is 0.15 ms/tick, 0.6% of the
rollout, and it is not worth doing.

Four separate errors of the same shape produced that, and each is worth more than the
rejected optimization.

## 1. A fixture whose distribution hid the property under test

My first measurement drew logits from a random normal. The nucleus came out at
**162301 of 248320** — so sorting the whole vocabulary looked like a 1.5x waste, i.e. no
finding at all. Peaked logits (`lg[:, :50] += 12.0`, what a trained policy produces) put the
nucleus at **43**, a 5776x waste.

The sharper statement is `tilerl-27`'s: random normal is not "unrealistic". It is a
distribution that happens to take the **worst possible value on the axis being measured**.
"Use realistic data" is the weaker rule; the useful one is *ask what your fixture's
distribution does to the quantity you are about to price*.

## 2. A ratio measured on the wrong hardware, when a bound was computable in advance

I priced a GPU mechanism with CPU numbers, and the ratio did not transfer: 12.4x on CPU,
2.26x on the card.

The bound was available before I ran anything. The logits are
`8 × 248320 × 4 B = 7.95 MB`. A radix sort makes a dozen or so passes over key and value —
about 0.16 GB of traffic — which at 4 TB/s is **~40 µs**. My CPU figure of 20.53 ms is
**517x** that. A number 500x above the target hardware's floor is not a measurement of the
target hardware.

The generalisation: **an O(V) operation whose V is only a few MB is compute-bound on a CPU
and bandwidth-bound on a GPU.** The ratio between them collapses, so a CPU-measured speedup
for that shape carries no information about the card. (48 derived the floor.)

## 3. A ruler in the wrong units, used to exclude the right answer

I excluded `.tolist()` as the cost by arguing: 27.2 s / 1024 ticks / 2 syncs = 13 ms per
sync, three orders of magnitude above a real device-to-host transfer, therefore not the
mechanism.

The arithmetic is right and the conclusion is wrong, because **the denominator was the wrong
quantity**. On an asynchronous stream `.tolist()` does not cost a transfer; it costs *waiting
for every outstanding kernel to drain*. That has no small upper bound. The whole decode path
from `engine.py:1025` to `:1347` contains no synchronize, so the forward's drain is billed to
the first host read — which is `_sample_commit`.

So the time was not *in* `.tolist()`; it was **exposed by** it. I used a ruler graduated in
transfer time to measure a queue, got "impossible", and excluded the correct answer with it.
Same shape as error 2: the ruler and the object were different quantities.

## 4. My own floor derivation reaches 0.73x of the measured value

48's argument that the forward timer is broken rests on a floor: the forward measured
3.07 ms/tick against a weight-stream floor of 6.11 ms/tick, i.e. **0.50x** — a forward cannot
complete in half the time needed to read the weights it multiplies.

I tried to derive that floor independently from `config.py:114`, and could not:

Every row labelled with where it came from, because that is what a reader cannot recover
later: 3.14 GB and 3.746 GB have the same unit, the same magnitude and the same number of
decimals, and nothing in the digits says which one is a dump.

| term | GB | source |
|---|---:|---|
| fp4 nibbles (0.5 B/param, incl. `lm_head`) | 12.55 | **derived** from param count |
| f32 block scales (one per 32 elements, `pack_fp4` block=32) | 3.14 | **derived** from block size |
| embedding table (bf16; a row lookup on decode, not streamed) | 2.54 | **derived** from vocab × hidden |
| **my total** | **18.23** | derived |
| **measured resident** (`wins/2026-09-03-grpo-27b-fits-the-card.md:49`) | **24.93** | **read** |

**0.73x, 6.7 GB unaccounted.** My first attempt was worse — 13.18–17.00 GB, having omitted
the block scales entirely. Had I judged 3.07 ms against *that*, I would have got 0.93x:
"close to the floor but not violating it", and **passed over a real instrument defect**.

The floor stands because the measured footprint is read, not derived. My derivation does not.

### The gap is located, and it is a term my derivation has no line for

`tilerl-48`'s per-key dump on card 1 (**read**, not derived — 24.433 GB against 24.437
resident):

| key | GB | what |
|---|---:|---|
| `wq` | 7.499 | fp4 nibbles |
| `scale` | 3.746 | fp4 block scales |
| `oscale` | 0.017 | |
| `w8` | 10.625 | **fp8 weights** |
| `wscale` | 0.003 | fp8 scales |
| `embed` | 2.543 | one row read on decode |

**Not every weight is fp4.** I assumed 0.5 B/param throughout and got 12.55 GB of nibbles;
measured is 7.499 GB of fp4 plus **10.625 GB of fp8 at 1 B/param — a row my table does not
have**. That single term is 159% of the 6.7 GB gap, and the remainder is my 5.05 GB nibble
over-count less a 0.61 GB scale under-count. So the gap was not a missing fraction of a term
I had; it was a whole quantization tier I did not know was there.

Two corrections that change nothing downstream, recorded because "it was wrong" and "it
mattered" are different claims. My 3.14 GB scale figure is **derived** and the measured value
is 3.746 — but that figure was an intermediate in this failed derivation, and the floor table
below takes its weight bytes from the dump, so correcting it moves no number. And 48's
`block=16` reading was itself back-derived from a 133.8/111.6 byte ratio rather than read off
`scale.shape`; measured 3.746 GB matches neither block=32's 3.14 nor a doubled 6.28, so the
shape is still unknown. **I nearly re-derived bytes on that unverified shape — the same error
a third time, arriving from another session.**

Two premises I could not verify, which bound how hard the 0.50x claim is: 4 TB/s is the
nominal bandwidth and achieved rates are typically 0.7–0.9 of it (which makes the violation
worse, not better); and I assumed each parameter is read exactly once per decode tick, which
I cannot rule out being reduced by caching or fusion. So the defensible statement is "3.07 ms
violates a floor computed from nominal bandwidth and one read per parameter", not "3.07 ms is
physically impossible".

### The weights-only floor was asked whether it flips, and it does not

`tilerl-27` asked the right follow-up: the 6.11 ms floor counts weights only, so if state,
KV and activations add enough traffic the utilization conclusion changes and the kernels are
already near the roof rather than 4x under it. Per decode tick at B=8:

Sources marked, per the rule above. The weight row is 48's per-key dump minus the embedding
(a decode reads one row, it does not stream the table); the rest is derived from the model
shape, since no dump exists for a transient.

| term | GB/tick | share | source |
|---|---:|---:|---|
| weights (fp4 + fp8 + scales, `lm_head` included, embedding excluded) | 21.896 | 93.8% | **read** (48's dump) |
| gated-delta recurrent state, read + write | 1.208 | 5.2% | derived |
| paged KV, mean over the run | 0.136 | 0.6% | derived |
| conv state | 0.047 | 0.2% | derived |
| activations | 0.042 | 0.2% | derived |
| logits | 0.008 | 0.0% | derived |
| **total** | **23.337** | | |

Floor **5.83 ms** at 4.00 TB/s nominal, **6.97 ms** at 3.35 TB/s achieved — 1.07x the
weights-only figure, not 3x. The KV term is small because GQA gives 4 kv heads, not 24. So
no flip: utilization is 20.4% and the headroom is **4.91x**. 27's alternative reading — a
floor near 20 ms, i.e. already at 70% of the roof — is refuted by the same table.

**The weights row is one addend of this total, not a competing estimate of it.** 48 proposed
substituting 21.896 for the total and withdrew it: doing so asserts a decode tick streams no
KV and no recurrent state, and at gen 1024 the KV plainly crosses the bus.

**A separate figure, and it must not be folded into this one.** 48 measured fp4 GEMM at
**1144.7 GB/s at M=8, 28.6% of nominal**. Dividing the same 23.337 GB by that rate gives
20.39 ms against a 28.64 ms measured forward — 71.2%. That is not "two thirds of the forward
is weight streaming": the rate is *itself* measured at M=8, so it already contains whatever
makes M=8 inefficient, and the ratio is close to an identity. The two rows answer different
questions and stay separate — how much hardware is left (nominal denominator: 20.4%,
**4.91x**), and how much the achieved rate fails to explain (measured denominator: 8.25 ms
unaccounted).

A weights-dominated model is the general case for a 27B at B=8, so "weights only" is a good
approximation here. That it *is* an approximation was worth checking rather than assuming,
because the check costs one table and the wrong answer costs a work programme.

The same table prices `tilerl-0a`'s B=16 finding, and the answer is that bandwidth does not
object. Only the per-batch terms double, so **23.337 → 24.778 GB, +6.2%, floor 5.83 → 6.19 ms**
— while the tokens produced double, i.e. **1.88x cheaper per token**. The KV doubling that
looks alarming is 0.136 GB on a 21.90 GB weight stream. So the cost of B=16 is the memory
pool, not the bandwidth, and the pool is a card measurement rather than a static assertion —
[measured, and it fits with 57 GiB spare](../wins/2026-09-08-b16-fits-and-three-predictions-were-wrong.md).

## 5. A self-consistent decomposition is not a correct one

`tilerl-48`'s own lesson from the card side, and it is the one that generalises furthest.
Their per-op table summed to the measured step every time, through three successive versions —
and the attribution was wrong in all three. Each fix exposed the next defect rather than
reaching the answer: the table was internally consistent while pointing at `_sample_commit`,
still consistent after the sync arm moved the time to the forward, and still consistent while
the forward timer itself read 3.07 ms against a 6.11 ms floor.

Arithmetic that closes is exactly the property that survives underneath a wrong attribution,
because a decomposition is built to sum to its total. The verdict has to come from outside the
partition — a floor, a control arm, an independent instrument — which is the same shape as
errors 2 and 3 in this entry: consistency inside one ruler says nothing about whether the
ruler measures the object.

The card's own numbers agree with the floor from outside: 48's sync arm reads 91.31 ms/tick,
which divided by the 3.14x eager-fallback confound gives 29.08 ms against the async total's
29.67 ms — **2.0%**. Two instruments that can fail differently, agreeing.

## The cross-validation that was not one

The two investigations were meant to be independent — 48 measuring on the card, me deriving
from the code. They were not: **we read the same source, so we reach the same conclusion by
construction.** One chain walked twice. (`tilerl-27` identified this and attributed the
mis-design to themselves.)

The actually independent arm is a control on the card: add one
`torch.cuda.synchronize()` after `_model.forward` and change nothing else, giving two
mutually exclusive outcomes — sampling stays at ~26.6 ms/tick, or the forward absorbs it and
sampling collapses to microseconds. That arm carries its own cost: it triggered
`cudaErrorStreamCaptureInvalidated` and fell back to eager, so it changes the path it
measures. Recorded rather than treated as clean.

## Rules

- **Ask what your fixture's distribution does to the quantity being priced.** A distribution
  can sit at the worst value on exactly the axis under test, and then the measurement reports
  that there is nothing to find.
- **Compute the target hardware's floor before measuring on other hardware.** A figure 500x
  above the floor is about the machine it ran on, not the machine it is meant to inform.
- **Check that the ruler and the object have the same units before excluding a suspect.** A
  sync on an async stream costs a drain, not a transfer; a sort's cost is compute on a CPU and
  bandwidth on a GPU.
- **A derivation that reaches 0.73x of a measurement cannot adjudicate a 0.50x claim.** State
  the gap; a floor is only as strong as the bytes you can actually account for.
- **A decomposition that sums to its total is not thereby correct.** It was built to sum. Three
  successive per-op tables all closed while the attribution was wrong; the verdict has to come
  from outside the partition.
- **Two agents reading one source are one evidence chain.** An independent check has to be
  able to fail differently.
- **Label every number as read or derived at the moment it is written.** 3.14 GB and 3.746 GB
  share a unit, a magnitude and a decimal count; nothing in the digits says which came from a
  dump. Provenance is unrecoverable later and it is what a reader needs first — twice today a
  derived byte count was 1.2x the measured one (48's 111.6 vs 133.8, my 3.14 vs 3.746), and a
  third instance was avoided only because its author noticed in the same sentence.
- **A rate measured at the operating point cannot explain that operating point.** Dividing by
  an achieved bandwidth prices the same inefficiency on both sides of the ratio, which lands
  near an identity. Nominal answers "how much hardware is left"; achieved answers "what does
  the achieved rate fail to explain". Never combine them into one figure.

## What is not rejected

The full-vocabulary sort **is** real waste — 43 tokens sampled out of 248320 sorted — and the
per-row `multinomial` loop is 9.35x its batched form on the card (0.790 vs 0.085 ms). Their
combined ceiling is about 0.85 ms/tick, ~2.6% of the rollout at 48's numbers. Not the 27 s,
and not worth doing until the drain accounting is settled and a step count exists to price it
against. The measurement stays on record so nobody re-derives it.
