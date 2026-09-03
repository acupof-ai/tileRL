---
question: Is the width-W verify tick lossless on the 27B, as speculation claims?
status: measured
source: H20 sm90, 27B NVFP4, `scripts/acc_spec_arms.py`, both arms in one process on one card
---

# W>1 speculation is not token-identical on the 27B; the split-KV NaN guard is

Speculation's claim is equality — greedy output under a width-W verify tick must
be the greedy output without one, token for token. `Engine._verify` states it
outright: *"an accepted token is bit-identical to the unspeculated one."* That
had been checked on the tiny model as a token-stream comparison and never on the
27B with an accuracy suite.

Two arms, one process, one card, through the same `eval.py` the training path
uses: no speculation, then W=8 (`spec_depth=7`) with the NextN draft head at
`$TILERL_QWEN38_SOURCE/model_mtp.safetensors`. Every completion string is kept
and diffed per question — two arms can score the same and still disagree, and
here they do.

## On current main: same score, different answers

`main` at `e26af2d`, GSM8K greedy, 8 questions, H20 GPU 5:

| | base | spec W=8 |
|---|---:|---:|
| score | 4/8 | 4/8 |
| **completions that differ** | | **4/8** |

The scores match and the answers do not. The divergences are coherent
alternative wordings, not corruption:

```
base  '**1. Calculate the total number of eggs available:**\nJanet's ducks lay 16 eggs per day.\n\n**2. Calculate the number of eggs used personally:** ...'
spec  '**1. Calculate the total number of eggs used personally:**\nJanet eats 3 eggs for breakfast and uses 4 eggs for muffins. ...'
```

```
base  '*   Each chicken requires 3 cups of feed per day.\n*   There are 20 chickens in the flock.'
spec  '*   Wendi has **20 chickens**.\n*   Each chicken requires **3 cups** of feed per day.'
```

W=8 genuinely engages here: all 70 wide ticks ran at width 8, 2751 drafts with
1334 accepted (48.5%), 22.78 tokens committed per decode forward against the
base arm's 6.77. So this is the verify tick working as designed and still not
reproducing the unspeculated stream.

Four of eight is a small sample and the mechanism is not established. A greedy
argmax can flip on a near-tie when the same position is computed by a different
kernel path (a W=8 verify tile versus a W=1 decode tile), which is float
nondeterminism rather than a logic error — but it is not what `_verify`'s
docstring promises, and these flips land early enough to change the whole answer
text. Either the claim comes down to "same distribution, not same tokens", or
the divergence is a bug; this run does not separate them.

## The split-KV NaN guard works, and here is the negative control

The first run of this gate was on `perf/spec-graph`, which is stacked on
`fix/attn-combine-head-dim` and **predates the split-KV NaN guard now on main**.
On that tree the same script gave:

| | base | spec W=8 |
|---|---:|---:|
| GSM8K greedy, n=500 | 196/500 = 39.2% | **0/500 = 0.0%** |
| completions that differ | | **500/500** |

Every speculative completion ran out as an unbroken tail of `!` — token id 0,
which is what `greedy` returns over all-NaN logits — and `torch.isnan` on the
trunk logits inside `_verify` fired on **372 of 560 verify row-ticks**. First-NaN
residues were enriched 3.7x at `n % 64` in `[1,7]` (28 of 69 against a null of
11%).

Same script, same model, same checkpoint, on `main`: **0 NaN row-ticks out of
393**, with all 64 residues reached and **39 of them at `n % 64` in `[1,7]`** —
the exact geometry the guard targets. The guard is

```python
scores_scale[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 1.0, ...)
acc_s[i, j]     = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0.0, ...)
```

in `make_paged_attention_decode`: a split whose first tile sits wholly past a low
chain row's causal bound leaves both softmax maxima at `-inf`, and
`(-inf) - (-inf)` is NaN.

A guard is worth what its negative control is worth. This is that control — the
same measurement on the tree without it, at the same widths, on the same
questions.

## What the broken tree also showed, and what it hid

On the pre-guard tree the configured W=8 barely ran: **70 wide ticks out of
14,398 decode forwards**, the rest at W=1. That is downstream of the corruption,
not independent of it — once a row's hidden is NaN, `DraftHead.confidence` is
NaN, every `survival` product is NaN, `p >= cut` is False for NaN, and
`verify_lens` keeps nothing, so a corrupted row decodes at W=1 for the rest of
its life. On `main` the same workload runs **70 of 70 wide ticks at W=8**.

That also means the residue histogram from the broken tree was a weaker
attribution than it looked: `_verify` only runs on wide ticks, so it was blind
to the 14,328 W=1 ticks, and a NaN created on a W=1 tick would only be observed
later at whatever residue the next wide tick happened to have.

## A standing lead, independent of the NaN

`LinearStatePool.alloc_slot` zeroes `states` and `conv_windows` and **does not
zero `step_states` or `step_windows`**. Those planes are written by a wide tick
and read back by `select_step`, and slots are reused across requests. Nothing
here triggered it, but an un-zeroed pool plane that a later request can read is
a bug waiting for a different trigger, and the asymmetry with the two planes
that *are* zeroed looks unintentional.

## Why the gates were green

**The tiny-model losslessness test cannot reach the geometry.**
`test_speculation_reproduces_greedy_decode` runs `prompt=[3,4,5,6], n=24`, so
the sequence length never exceeds 28. `block_N` is 64: only split 0 is ever
non-empty, and the wholly-masked split never exists. It passes on sm90 and could
not have caught the NaN. It also cannot catch the divergence above, because a
tiny random model has no near-ties worth flipping.

**MMLU 0-shot exercises no speculation at all**, so it is not a control.
`mmlu_score` runs at `max_new_tokens=1`: the answer letter comes off the
request's prefill forward and the request finishes there. Measured on the 27B,
1000 questions, both arms: `decode_forwards = 0`. In a warm 200-question re-run
both arms report `decode_forwards 0, prefill_forwards 113, spec_drafted 0` — the
draft head does not even draft, because `_draft_step` skips a row already in
`_PHASE_DONE`. The two arms execute identical code and agree 1000/1000. That
agreement is a tautology, not evidence.

The 377.8 s the spec arm spent on MMLU against the base arm's 94.8 s is
first-touch JIT of the draft path, not a per-request cost: warm, the same
200-question comparison on one card is 38.2 s base against 32.1 s spec.

## Cost

The draft head and its step planes cost **+12.3 GiB** on `main` (37.72 vs 25.43
GiB peak). No usable end-to-end throughput number came out of either run — every
spec phase contained first-touch JIT with the GPU at 0% for long stretches, and
on the broken tree the completions were garbage that never hit EOS. The
~921 tok/s projection remains unconfirmed and is not confirmable as stated: it
multiplies a graph-replay tick by DFlash2's acceptance of 5.80, and DFlash2 is
"not on the engine tick yet".

## Rule

A speculative arm is not verified by two matching accuracy percentages. Diff the
completion strings — on `main` the two arms both score 4/8 and disagree on 4 of
8 answers, and the score alone would have called that a pass.

Measure the tree you mean to measure. The first run of this gate was on a branch
stacked one commit below the fix for the exact failure it found, and the result
read as a live catastrophic bug instead of a confirmation that a guard works.
Name the commit in the entry, not the branch.

An eval that generates one token per question exercises no decode tick and no
draft. Check `decode_forwards` and `spec_drafted` before believing a suite is a
control — a green check on a code path that never ran is worse than no check.
