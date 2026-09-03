---
question: What does block-parallel drafting buy on sm70, once the draft/verify split is measured rather than fitted?
source: V100 32GB sm70, ctx 1024, B=1, Qwen3.8-27B NVFP4 + shipped 1-layer MTP head. Arithmetic over the measured staircase in wins/2026-09-02-draft-is-two-thirds-of-a-spec-tick.md; DSpark decay shape from a 6-agent source study (wf_09315e25-5e3).
---

# Block-parallel drafting: 1.016x, and the one thing it buys is negative here

**Verdict: reject, and the reason is not the ceiling — it is that the mechanism's
own advantage costs more than it returns on this card.**

## The gating number, and what it licenses

Measured (not fitted — depths 2 and 3 share rung 4, so the difference is exactly
one draft forward):

| | |
|---|---:|
| one draft forward | **5.53 ms** |
| depth-3 tick | 66.46 ms |
| **draft share** | **25.0%** (3 x 5.53) |
| tok/forward | 3.34 |
| tok/s | 50.3 |

Three ceilings, and only one of them is a fair comparison:

| scenario | tick | ceiling | tok/s |
|---|---:|---:|---:|
| draft cost → **0** (not achievable by anything) | 49.87 | 1.333x | 67.0 |
| **one** draft forward (what block-parallel actually does) | 55.40 | **1.200x** | 60.3 |
| one forward, DSpark's measured suffix decay | 55.40 | **1.016x** | 51.1 |

The middle row is the honest ceiling for the mechanism: a block head replaces D
sequential forwards with **one**, not with zero. Break-even at that tick is
**2.784 tok/forward** — below it the parallel head loses to the head we ship.

## Why it fails: the decay eats the whole margin

Our per-position acceptance is not read off a probe, it is inverted from the
measurement: solving `1 + p + p² + p³ = 3.34` gives **p = 0.881** (reproduces
3.340, so the uniform model is not a distortion at this depth).

A parallel position cannot see what was sampled before it. Applying DSpark's own
measured decay *shape* (its `[72, 57, 45]` → ratios 0.79, 0.79) to our p:

    0.881, 0.696, 0.550  →  2.831 tok/forward

**2.831 against a 2.784 break-even.** The margin is 1.7%, against a harness noise
floor of 1.16% — the win is the same size as the instrument. That is the verdict:
not "too small to matter" but "indistinguishable from zero at this depth".

## The structural gift is negative on sm70

Block-parallel's real gift is that **draft cost is O(1) in block size**, so wider
blocks are free — filling rung 8 needs 7 sequential forwards (38.7 ms) with our
head and one (5.53 ms) with a block head. That is what the H20 3.64x ceiling
(`wins/2026-09-03-drafter-batching-ceiling.md`) is about, and it is a different
question: batching the drafter **across rows** at B=8, not across positions.

Spending the gift here, one forward at every k, against the measured verify
staircase {rung 2: 36.58, rung 4: 49.87, rung 8: 68.46}:

| k | W | rung | tick | tok/fwd | tok/s | vs 50.3 |
|--:|--:|--:|-----:|--------:|------:|--------:|
| 2 | 3 | 4 | 55.40 | 2.494 | 45.0 | 0.895x |
| **3** | **4** | **4** | **55.40** | **2.831** | **51.1** | **1.016x** |
| 4 | 5 | 8 | 73.99 | 2.977 | 40.2 | 0.800x |
| 5 | 6 | 8 | 73.99 | 3.027 | 40.9 | 0.813x |
| 6 | 7 | 8 | 73.99 | 3.041 | 41.1 | 0.817x |
| 7 | 8 | 8 | 73.99 | 3.044 | 41.1 | 0.818x |

Positions 4-7 add **+0.213 tok/forward** and cost **+18.59 ms** of rung-8 verify.
Going wider — the only thing block-parallel makes cheap — takes the arm from
1.016x to **0.818x**. The optimum is k=3, which is the depth we already run.

So the mechanism's advantage has nowhere to land: the sm70 GEMV rung ladder
{1,2,4,8,32} prices verify in steps, and the step from rung 4 to rung 8 is
larger than four positions of a decaying suffix are worth.

## The gate, with its negative control

The table above is arithmetic over `LADDER_WIDTHS` and the measured staircase, so
it runs on any target and lives next to both, in `spec.py`'s `__main__` check.
Three assertions: k=3 stays the optimum, k=7 stays below it, and k=3's margin
stays inside the 1.16% noise floor.

Negative control **run, not asserted**: price rung 8 at rung 4's 49.87 ms — the
"wider is free" world the mechanism assumes — and the optimum moves to **k=7 at
54.9 tok/s**, tripping the first two. A mutation that only *removed* rung 8 does
not discriminate, because it removes the k=4..7 rows along with the price; the
mutation has to keep the rows and change the cost.

## What is confirmed, and what stays unknown

Confirmed by a 6-agent source study, independently of this arithmetic:

- **DSpark/DFlash is genuinely block-parallel.** Four shape facts force it —
  `fc.weight [5120, 25600] = 5 x 5120` is a per-*request* context projector at
  one hidden width; **zero per-position parameters** across all 63 tensors; K/V
  is `cat(context, noise)` with the noise block bidirectional; the config carries
  a scalar `block_size 7`, not a step count. `agent-infer`'s
  `qwen35/dspark.rs:919-921` fills `ids` with `mask_token_id` and overwrites row
  0 with the anchor.
- **The port is not the 1-layer head we ship.** DSpark's is 5 layers / 1.86B
  against our 1 layer / 456M (`spec.py`: `num_layers=len(idx) or 1`). Its single
  forward is of a 5x deeper network, so it plausibly costs more than the three it
  replaces — which would put the arm below parity. Unmeasured, and it does not
  need measuring: the 1.7% margin above already fails before this term is added.
- **agent-infer measured a DSpark head at 13% acceptance on Qwen3.8**, cause
  recorded as unexplained. That is the largest unpriced risk in a port.

Named unknowns, not filled with guesses: whether the 5-tap context KV refreshes
with newly accepted tokens during decode (no primary source states the append
rule), and the recurrent-state rollback contract for the 48 gated-delta layers.

## Two corrections this pass made

**`docs/roadmap.md` said "on sm70 the draft is in-graph". It is not.**
`_run_decode_graph` captures only `model.forward` (`engine.py:231,237`); the
draft runs at `engine.py:908`, **after** `g.run()` returns, and `DraftHead.step`
allocates numpy arrays whose shapes (`w`, `nb`, `n`) are data-dependent per tick
— which is exactly why it cannot be captured. Neither source entry ever claimed
in-graph; the roadmap line was written from memory of the wrong entry. Capturing
it is separately rejected at a 1.14x ceiling
(`errors/2026-09-02-capturing-the-draft-is-rejected.md`).

**A ceiling for "cost → 0" is not a ceiling for the change being priced.** The
1.333x free-drafter figure appears nowhere as a recommendation, but it is 1.11x
larger than the 1.200x the mechanism can reach, and reading the wrong one is how
a 1.7% margin looks like a 33% one.

## Rule

Price the mechanism, not the absence of the cost. "Draft share is 25%" bounds
*any* change to drafting at 1.333x, but a block head does one forward rather than
none, so its own bound is 1.200x — and the gap between those two is larger than
the whole remaining margin.

Second: when a mechanism's advantage is "X becomes cheap", check what X costs on
the target before pricing the advantage. Here X is verify width, the sm70 ladder
prices it in steps of 1.37x, and the suffix decays faster than the step — so the
advantage is worth less than zero, which no amount of draft-side saving fixes.

## 2026-09-03, later: the margin was thinner than stated, and the verdict holds

The `p = 0.881` above was inverted from `tok/forward = 3.34`, measured on a prompt
that was `range(10, 10+ctx)` — consecutive low token ids, which the draft finds
unusually easy. On a prompt drawn from the whole vocabulary the same config reads
**2.03 tok/forward** at ctx=1024 (`wins/2026-09-03-long-context-decode-is-all-tick-cost.md`),
which inverts to **p = 0.554**.

Re-running this entry's own arithmetic on that:

| acceptance source | tok/fwd | p | block-parallel yield | vs 2.784 break-even |
|---|---:|---:|---:|---:|
| confounded prompt (above) | 3.34 | 0.881 | 2.831 | **1.017x** |
| random-vocabulary prompt | 2.03 | 0.554 | 1.880 | **0.675x** |

So the REJECT stands and its margin widens from 1.7% to 32%. That matters for how
much scrutiny the verdict needs, not for its direction: at 1.016x against a 1.16%
noise floor this was one careful re-measurement away from flipping, and it is now
outside any noise question.

The draft-share half of the measurement is unaffected. `share = 2 × (ms_tick(3) −
ms_tick(2)) / ms_tick(2)` is a cost subtraction at a fixed rung, and tick cost does
not depend on which tokens are in the prompt.

Neither prompt is the serving distribution — random vocabulary is the pessimistic
end and consecutive low ids the optimistic one. **The depth default is therefore
still unsettled**, and settling it needs real text; this entry's verdict does not
depend on which end is right, because the parallel head loses at both.

