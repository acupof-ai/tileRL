---
question: Is the width-W verify tick lossless on the 27B, as speculation claims?
status: measured
source: H20 sm90 GPU 7, 27B NVFP4, both arms in one process on one card
---

# The W=8 verify tick is not lossless on the 27B: it returns NaN, and NaN decodes to `!`

Speculation's whole claim is equality — greedy output under a width-W verify
tick must be the greedy output without one, token for token. That had been
checked on the tiny model as a token-stream comparison and never on the 27B
with an accuracy suite.

Two arms, one process, one card, through the same `eval.py` the training path
uses (`scripts/acc_spec_arms.py`): no speculation, then W=8 (`spec_depth=7`)
with the NextN draft head at `$TILERL_QWEN38_SOURCE/model_mtp.safetensors`.
Equality is the test, so every completion string is kept and diffed per
question — two arms can score the same and still disagree.

## The finding

27B NVFP4, H20 GPU 7, one process, `decode_graph=False`,
`prefix_store=NoPrefixStore()`, concurrency 8 — the training path's own engine
config, matching the P1 baseline of 2026-09-02.

| suite | base | spec W=8 | completions that differ |
|---|---:|---:|---:|
| MMLU 0-shot | 742/1000 = 74.2% | 742/1000 = 74.2% | 0/1000 |
| GSM8K greedy | 196/500 = 39.2% | **0/500 = 0.0%** | **500/500** |

**Every one of 500 GSM8K completions differs, and the speculative arm answers
nothing correctly.** Each spec completion runs out as an unbroken tail of `!`.

```
q1 base  'To determine the total number of bolts required, we can break down the requirements for each type of fiber:\n\n1.  **Blue fiber**: ...'
q1 spec  'To determine the total number of bolts required, we can break down the requirements for each type of fiber:\n\n1.  **!!!!!!!!!!!!!!!!!'
```

`!` is token id 0 in this checkpoint's tokenizer (`decode([0]) == '!'`,
`encode('!') == [0]`), and `greedy` over all-NaN logits returns index 0. So a
tail of `!` is a NaN logit row, printed — and `torch.isnan` on the trunk logits
inside `_verify` confirms it directly: **372 of 560 verify row-ticks came back
NaN**.

The base arm reproduces the P1 baseline — MMLU 742/1000 exactly, GSM8K 196/500
against 194/500 — which is what says this is the same harness.

## The configured W=8 almost never ran

The width histogram over the GSM8K phase is the most important number here:

| tick width | 1 | 2 | 4 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| decode forwards | 14328 | 15 | 1 | 4 | 3 | 47 |

**70 wide ticks out of 14,398.** 99.5% of decode ticks ran at W=1. Drafted
3080, accepted 2066 across 500 requests generating ~115k tokens.

That is self-consistent with the corruption rather than independent of it: once
a row's hidden state is NaN, `DraftHead.confidence` is NaN, every `survival`
product is NaN, `p >= cut` is False for NaN, so `verify_lens` keeps nothing and
the row decodes at W=1 for the rest of its life. The wide ticks are the
pre-corruption window.

## What the residues do and do not show

The tracer records `n % 64` for `SeqLens = seq_len - 1 + W` at every wide tick,
and separately at every tick whose trunk logits came back NaN.

- All 64 residues were reached (560 wide row-ticks), so the run had coverage.
- Of the 69 requests whose **first** NaN was caught inside `_verify`, **28 sit
  at `n % 64` in [1,7]** — 41% where the null expectation is 7/64 = 11%, an
  enrichment of 3.7x. That is consistent with the derived wholly-masked-split
  condition.
- It is **not** proof that the split is the only cause. `_verify` runs only on
  wide ticks, so the tracer is blind to the 14,328 W=1 ticks; a NaN created on a
  W=1 tick is only *observed* later, at whatever residue the next wide tick
  happens to have. And 500 requests are corrupted while only ~560 wide row-ticks
  exist in the whole run — roughly one per request — so corruption reaching
  every request cannot be read off the wide ticks alone. Either NaN also arises
  on the W=1-with-draft path, or it propagates between requests through state
  the pool reuses (`step_states` / `step_windows` are not zeroed by
  `alloc_slot`, unlike `states` and `conv_windows`).

Both remain open. What is settled is that attaching this draft head at
`spec_depth=7` corrupts every GSM8K answer on the 27B.

## Speed, since the accuracy row and the speed row come from one session

| arm | GSM8K wall | tok/s | tok per decode **forward** | peak |
|---|---:|---:|---:|---:|
| base | 1355.6 s | 86.9 | 7.94 | 25.90 GiB |
| spec W=8 | 4745.9 s | 25.3 | 8.07 | 37.46 GiB |

The speculative arm is **3.4x slower end to end**, not faster. It is not a fair
speed comparison — every request runs to the 256-token cap emitting `!` instead
of stopping at EOS, and the phase contains first-touch JIT with the GPU at 0%
for long stretches — but it is the only end-to-end number, and the projection it
was supposed to confirm was ~921 tok/s.

`tok per decode forward` is tokens committed per decode *forward*, not per row:
at concurrency 8 an unspeculated forward commits one token for each running row,
so 7.94 says the batch held ~8 rows. It is not a speculation win, and the spec
arm's 8.07 is not one either.

The draft head costs **+11.56 GiB** (37.46 vs 25.90 GiB peak) for the step
planes and its own weights.

## The same failure on this branch alone

The 500-question run above was measured on this branch merged with
`fix/noprefix-snapshot-leak`, because the unfixed tree retains one 149.6 MiB
state snapshot per 16 generated tokens under `NoPrefixStore` and cannot finish a
500-question GSM8K eval on one card. The control is a smoke run on **this branch
with nothing else merged in**: 8 GSM8K questions, base 4/8, spec 0/8, 8 of 8
completions differ, every spec completion ending in `!`. Divergence runs from
char 19 to char 267 across the eight, with no pattern in the question — a
length-dependent condition, each request meeting it at a different point in its
own decode. Three of the eight commit different *sane* text first and collapse
later, so a NaN tick can commit a wrong-but-plausible token before the state is
poisoned.

## Why the gates were green

**The tiny-model losslessness test cannot reach the geometry.**
`test_speculation_reproduces_greedy_decode` runs `prompt=[3,4,5,6], n=24`, so
the sequence length never exceeds 28. `block_N` is 64: only split 0 is ever
non-empty, and the wholly-masked split the bug lives in never exists. It passes
on sm90 and is worth nothing here. The fix needs a case whose sequence length
crosses a `block_N` boundary at W>1.

**MMLU 0-shot exercises no speculation at all**, so it is not a control.
`mmlu_score` runs at `max_new_tokens=1`: the answer letter comes off the
request's prefill forward and the request finishes there. Measured on the 27B,
1000 questions, both arms: `decode_forwards = 0`. And in a controlled
200-question re-run, both arms report `decode_forwards 0, prefill_forwards 113,
spec_drafted 0` — the draft head does not draft either, because `_draft_step`
skips a row already in `_PHASE_DONE`. The two arms execute the same code and
agree on 200/200 completions; that agreement is a tautology, not evidence.
An MMLU that tested speculation would have to generate more than one token.

The 377.8 s the spec arm spent on MMLU against the base arm's 94.8 s is not a
per-request draft cost — it is first-touch JIT of the draft path, which the spec
arm pays because it is the first phase to touch those kernels. On a warm cache
the same 200-question comparison on one card is 38.2 s base against 32.1 s spec,
i.e. no penalty and no difference beyond noise, which is what a workload
executing identical code should look like.

## Rule

A speculative arm is not verified by two matching accuracy percentages. Diff
the completion strings; equality is the claim, and a suite average hides
per-question disagreement in both directions.

An eval that generates one token per question exercises no decode tick and no
draft. Check `decode_forwards` and `spec_drafted` before believing a suite is a
control — a green check on a code path that never ran is worse than no check.

When a kernel's behaviour depends on sequence length, gate it at lengths that
straddle its tile boundary. The losslessness test lived entirely inside the
first `block_N`, so it exercised one split out of the two the bug needs.
