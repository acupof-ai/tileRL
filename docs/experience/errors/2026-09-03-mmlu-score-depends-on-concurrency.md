---
question: The same MMLU slice scored 746/1000 and 742/1000 on clean main. What moved?
status: MECHANISM CORRECTED 2026-09-03 -- concurrency moves 0 of 1000 answers; the real variable was fuse_projections (see 2026-09-03-unfused-prelude-double-rounds.md). The recording fix stands.
source: read on origin/main 87f59b2; the 4-question gap measured on H20 sm90, 27B NVFP4
---

# An MMLU score depends on concurrency, and nothing recorded which one was used

Two runs of the same slice on clean main returned **746/1000 = 74.6%** and
**742/1000 = 74.2%**. This is not the moved-slice bug from earlier the same day:
`mmlu_accuracy(engine, tok, n, seed=0, ...)` passes `seed` straight to
`mmlu_questions(n, seed)`, so the draw is pinned. The slice is the same 1000
questions.

The difference between the two runs was **concurrency**: 32 (the
`mmlu_accuracy` default, which `scripts/mmlu.py` took) against 8 (what
`cli.py:197` passes).

## Why concurrency can change an answer

**This section is wrong, and the correction is measured.** Concurrency 8 and 32
return 742/1000 each with **0 of 1000 answers differing**, so concurrency changes
no MMLU answer. `M` on a prefill tick comes from `_PREFILL_BUCKET` (64) and the
prompt's own padded length, not from the batch: concurrency batches independent
rows and each keeps its own width. `M = B * W` is a **decode**-tick mechanism, and
MMLU at `max_new_tokens=1` runs no decode ticks. The variable that actually
differed between the two callers was `fuse_projections` —
[the unfused prelude double-rounds](2026-09-03-unfused-prelude-double-rounds.md).

The fix recorded below still stands: recording which concurrency a score used was
right, and the reason given for why it mattered was not. Kept rather than deleted
because the reasoning is the error worth reading.

`concurrency` sets the engine's batch size. The batch size sets `M = B * W`. And
`M` picks the fp4 linear arm:

```
_MGEMV = 3      2 <= M <= 3  -> linear_fp4_gemv at factory M
_MX    = 8      2 <= M <= 8  -> linear_fp4_mma8, padded to 8 rows
                     M > 8   -> quant_fp8 + w4a8 WGMMA at _snap_mma_tile(M, 128)
```

So two concurrencies can run **two different kernels, with two different
reduction orders, on the same question**. Wherever the top-2 logit gap for the
answer letter is below the arm-to-arm difference, the argmax flips and the letter
changes.

The width-ladder measurement on the same card puts a number on that difference:
changing the linear arm moves the top-1 logit by a **median 0.153** (~1.6 bf16
ulps at logit magnitude 24, because `Backend._rows` re-casts every activation to
bf16 once per layer), while holding the arm fixed gives **exactly 0.000** over
1708 shared positions. 4 flips in 1000 four-way-choice questions is the size of
effect that produces.

`max_new_tokens=1` does not protect against this. Every MMLU answer comes off
the prefill forward, but the prefill still runs through `M`-dependent arms.

## What is fixed here

Not the flips — the **silence**. A score whose value depends on a knob has to
carry the knob, the same rule the slice got when it moved under a number earlier
today.

- `mmlu_accuracy` returns `(correct, total, concurrency)`.
- `cli.py` records `mmlu_<tag>_concurrency` in the manifest and prints it beside
  the score.
- `scripts/mmlu.py` pins `CONCURRENCY = 8` — one value, matching `cli.py`, so
  the two callers no longer disagree — and writes `seed` and `concurrency` into
  its JSON.

`test_mmlu_score_reports_the_concurrency_it_used` gates the contract, not the
number: `mmlu_accuracy` must return the concurrency and `cli.py` must unpack and
record it. Gated on the signature because `mmlu_accuracy` needs the real dataset,
and what regresses is a caller unpacking two values again. Reverting `cli.py` to
the two-tuple turns it red, which was run rather than assumed.

## What is not settled

**Settled 2026-09-03, and the answer was that the question was wrong.** There are
no 4 flipped questions: concurrency 8 and 32 both score 742/1000 and **0 of 1000
answers differ**, with 41 questions (4.1%) sitting under the 0.153 arm-change
delta — near-ties existed and none of them moved. The 746-vs-742 gap was
`fuse_projections`, not concurrency, and the flipped answers there are **not**
near-ties: 53 questions, |Δ logit| median 1.071 and max 4.462, only 3 of 53 inside
0.153. That is the wide-gap case this section called a defect, and it was one —
[the unfused prelude double-rounds](2026-09-03-unfused-prelude-double-rounds.md).

So **74.6% and 74.2% are two different numbers, not one measured two ways**: 74.6%
is the accurate arm.

## Rule

If a number's value depends on a parameter, the number is not reportable without
the parameter. This is the second instance in one day — the first was the eval
slice moving under a score — and both were invisible for the same reason: the
parameter had a default, and two callers took different ones.

A batch-size knob is a numerics knob on this backend, not just a throughput
knob. `M` selects the kernel, so anything that moves `M` can move an answer.
