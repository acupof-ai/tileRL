---
question: The same MMLU slice scored 746/1000 and 742/1000 on clean main. What moved?
status: mechanism identified and the recording fixed; the 4 questions are not yet attributed
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

**Whether those 4 questions are near-ties.** The mechanism above predicts they
are, and the ladder makes the prediction quantitative, but nobody has looked at
their top-2 gaps. The discriminator is the same one the W=8 divergence used: a
flip at a gap below ~0.15 is arithmetic; a flip at a wide gap is a defect and
matters far more than 0.4%. Same slice, same seed, same card, concurrency 8
against 32, comparing per-question top-2 margins.

Until that runs, **74.6% and 74.2% should be read as one number measured two
ways**, not as a change.

## Rule

If a number's value depends on a parameter, the number is not reportable without
the parameter. This is the second instance in one day — the first was the eval
slice moving under a score — and both were invisible for the same reason: the
parameter had a default, and two callers took different ones.

A batch-size knob is a numerics knob on this backend, not just a throughput
knob. `M` selects the kernel, so anything that moves `M` can move an answer.
