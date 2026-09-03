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

**GSM8K greedy, 8 questions, PR #12's own tree with nothing else merged in:
base 4/8, spec 0/8, 8 of 8 completions differ.** Every spec-arm completion ends
in an unbroken tail of `!`.

| q | base chars | spec chars | diverges at char |
|---:|---:|---:|---:|
| 0 | 624 | 409 | 123 |
| 1 | 423 | 345 | 115 |
| 2 | 903 | 469 | 267 |
| 3 | 541 | 309 | 69 |
| 4 | 829 | 270 | 19 |
| 5 | 845 | 410 | 117 |
| 6 | 850 | 282 | 32 |
| 7 | 894 | 427 | 180 |

```
q3 base  'To find the total number of meters James runs in a week, we can break the problem down into steps:\n\n**Step 1: ...'
q3 spec  'To find the total number of meters James runs in a week, we can break!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
```

`!` is token id 0 in this checkpoint's tokenizer (`decode([0]) == '!'`,
`encode('!') == [0]`), and `greedy` over all-NaN logits returns index 0. A tail
of `!` is a NaN logit row, printed.

Three of the eight (q0, q5, q7) diverge into different but *sane* text first
and collapse to `!` later, so a NaN tick can commit a wrong-yet-plausible token
before the state is poisoned. Divergence runs from char 19 to char 267 with no
pattern in the question — what a length-dependent kernel condition looks like,
each request meeting it at a different point in its own decode.

## Root cause

Derived independently in review of PR #12 and confirmed by this run.
`make_paged_attention_decode` splits the KV range. At W>1 a split whose tile
range is non-empty can still be **wholly masked** for the low chain positions:
`scores_max` stays at `-inf`, `exp2(-inf - (-inf))` is NaN, and the combine's
`0 * NaN` propagates it instead of discarding an empty slice. The condition is
`n % 64` in `[1, W-1]` — 7 of every 64 sequence lengths at W=8.

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

## Numbers from the run

27B NVFP4, H20 GPU 7, one process, `decode_graph=False`,
`prefix_store=NoPrefixStore()`, concurrency 8 — the training path's own engine
config, matching the P1 baseline of 2026-09-02.

| arm | MMLU 0-shot | decode fwd | GSM8K greedy | wall | tok/s | tok per decode forward |
|---|---:|---:|---:|---:|---:|---:|
| base | 742/1000 = 74.2% | 0 | 196/500 = 39.2% | 1355.6 s | 86.9 | 7.94 |
| spec W=8 | 742/1000 = 74.2% | 0 | see above (8-question smoke) | | | |

The base arm reproduces the P1 baseline: MMLU 742/1000 exactly, GSM8K 196/500
against 194/500. **`tok per decode forward` is tokens committed per decode
*forward*, not per row** — at concurrency 8 an unspeculated forward commits one
token for each running row, so 7.94 says the batch held ~8 rows, which is the
number being correct, not a speculation win.

Memory, measured separately at 200 MMLU questions on one card: base peak 25.50
GiB, W=8 peak 37.39 GiB. **The W=8 step planes and the draft head cost
+11.89 GiB.**

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
