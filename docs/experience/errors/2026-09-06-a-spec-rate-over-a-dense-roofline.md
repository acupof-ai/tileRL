# A speculative rate divided by a dense roofline — 2026-09-06

> Status: open on one operand. The four wrong denominators are fixed here; the
> speculative ceiling stays a **32-42% bracket** because a draft step's streamed
> bytes are bucketed nowhere. Listed in OPEN.md.

## Context

Four sites cite the retired `64 tok/s` weight roofline as a live denominator. The
obvious repair is to swap in the current figure. Doing the arithmetic first
found the wrong defect: the denominator's **value** is the smaller problem, and
updating it alone moves every conclusion in the wrong direction.

## Root Cause

**The numerator is a speculative rate and the denominator is a dense
single-token roofline.** A weight roofline is bytes-streamed-per-forward over
bandwidth, so it bounds **forwards per second**. Speculation exists to emit more
than one token per forward — 2.95 tok/forward measured beside the 52.7 — so a
speculative tok/s is not the quantity that bound applies to. Dividing them
compares a rate against a ceiling that is `tok/fwd` times too low.

The correction is one identity, and it needs no byte count at all:

```
% of the speculative ceiling  =  (% of the dense roofline) / (tok per forward)
```

Both sites, with the roofline that was current **when each was measured**
(16.04 GB → 56.1 tok/s; the 14.44 GB → 62.3 f16-scale figure postdates them by a
day, so using it would be a second era error):

| site | published | numerator is | tok/fwd | era-correct vs dense | **vs the speculative ceiling** |
|---|---|---|---|---|---|
| `wins/2026-09-01-sm70-gemv-packed-x-f16.md:83` | 52.7 is **82%** of 64 | depth 3 | 2.95 | 94% | **32%** |
| `wins/2026-09-01-sm70-attention-thread-redundancy.md:93` | 46.5 is **73%** of 64 | depth 3 | 2.90 | 83% | **29%** |
| `LOG-v100-sm70.md:211` | 52.7 is **82%** of 64 | depth 3 | 2.95 | 94% | 32% |
| `CHANGELOG.md:333` | 52.7 is **82%** of 64 | depth 3 | 2.95 | 94% | 32% |

**Fixing only the stale value makes it worse.** Against the f16-era 62.3, 52.7
reads **85%** and 46.5 reads **75%** — both *higher* than the figures they
replace, so the repair that looks like diligence publishes a stronger claim on
the same broken comparison. That is what makes this worth an entry rather than a
sed: the stale-citation framing predicts a correction downward, and the tell
that the framing is wrong is that the correction goes up.

**The upper bound is what is claimable; the exact share is not.** 32% assumes the
draft forwards stream **zero** bytes, which is false — it is the ceiling's
ceiling, so 32% is an upper bound on the share and the true share is lower.
Charging the draft head plus lm_head at f32-era bytes for all three draft steps
(3 × (0.85 + 0.80) = 4.95 GB, from
[roofline is the streamed subset](2026-09-02-roofline-is-the-streamed-subset.md))
gives 126.5 tok/s and 42%. The two ends bracket the answer at **32–42%**, and
both reject 82%. Which end is right depends on the draft's streamed bytes, which
nothing in the tree measures — the 0.85 GB is the head's *resident* size, and a
draft step's streamed subset has never been bucketed the way
`check_scale_f16.py` buckets the trunk's.

## Four other citations of 64 are correct and are not touched

The same grep returns four more, and the criterion that separates them is
whether the number is used as a **live** denominator:

| site | why it stands |
|---|---|
| `wins/2026-08-31-sm70-split-kv-decode-attention.md:96` | already says it "originally cited a remembered 14 GB / 64 tok/s" and links the correction |
| `wins/2026-08-29-sm70-volta-fp4-cell.md:90` | reports that day's *estimate*, immediately followed by "Measured 19.9" |
| `errors/2026-09-01-spec-warmup-one-width.md:35` | a sanity check — 289 tok/s exceeded it, and a dense bound rejects 289 at any of these values |
| `scripts/bench_b1_decode.py:50` | the same check, in the comment that keeps the warmup honest |

The last two are the interesting pass: they divide a *speculative* 289 by a dense
bound too, and the comparison survives anyway because a bound only has to be
exceeded to fire. A one-sided test tolerates a denominator an equality cannot.

## Fix

Replace the four live sites with the ceiling the numerator belongs to, stated as
a bracket, and keep the era-correct dense figure beside it since the dense
comparison is the one the kernel work was judged on. No runtime change.

## Rule

Before dividing by a roofline, check that the numerator counts the same events
the roofline bounds. A weight roofline bounds forwards; speculation, batching,
and any form of multi-token emission put a factor between forwards and tokens,
and that factor is exactly the overstatement.

Second, from how this was nearly mis-repaired: **a stale operand and a wrong
operand need different fixes, and the stale framing hides the wrong one.**
"Update the number" is a mechanical edit that never re-derives the comparison, so
it cannot find a dimensional error — and here it would have raised 82% to 85%
while looking like a correction. When a repair moves a number *away* from
conservative, stop and re-derive instead of committing it.

Third: a bound used one-sidedly can survive a denominator that an equality
cannot. Do not read the four surviving citations as evidence the number was fine.

## Results

No runtime change. Documentation-only; exempt from the bench gate.

| date | commit | quantity | published | corrected |
|---|---|---|---|---|
| 2026-09-06 | (this) | depth-3 52.7 tok/s vs its own ceiling | 82% of 64 dense | **32–42%**, 94% of era dense |
| 2026-09-06 | (this) | depth-3 46.5 tok/s vs its own ceiling | 73% of 64 dense | **29–37%**, 83% of era dense |
| 2026-09-06 | (this) | draft step streamed bytes | — | **unmeasured** — the bracket's width |
