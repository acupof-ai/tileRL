# verify_lens is not dead — it binds at depth 6-7 with a cost model of the wrong shape — 2026-09-03

> Status: measured, recorded, **not changed**. The fix is a design change, not a
> constant, and choosing it needs a goodput measurement.

## Context

[verify-lens-is-dead-and-its-cost-model-is-wrong](2026-08-30-verify-lens-is-dead-and-its-cost-model-is-wrong.md)
records that the policy "keeps everything, exactly as the constants imply" and
calls it "a policy that runs and always returns its maximum." That reading was
about to justify deleting `survival`, `verify_lens`, `BIAS_MS` and `ROW_MS` —
roughly 40 lines — as provably inert.

Its evidence is a tiny-cell table at `spec_depth` 1, 2 and 4.

## The 27B disagrees, and the old table is the reason it looked inert

NextN head, `spec_depth=7`, 24 GSM8K prompts, B=8, graph on, H20:

```
verify_lens calls (per row): 1334
trimmed (kept < offered)   : 235  = 17.62%

  offered 1  kept 1  x19        offered 6  kept 4  x3
  offered 2  kept 2  x22        offered 6  kept 5  x5
  offered 3  kept 3  x28        offered 6  kept 6  x50
  offered 4  kept 4  x37
  offered 5  kept 5  x51        offered 7  kept 1  x1     kept 5  x73
                                offered 7  kept 2  x6     kept 6  x114
                                offered 7  kept 3  x9     kept 7  x892
                                offered 7  kept 4  x24
```

**Zero of 157 trimmed at offered 1-5. 8 of 58 at offered 6. 227 of 1119 at
offered 7.** The policy is inert below 6 and binds above it.

Depths 1, 2 and 4 are exactly the region where it cannot bind. The old claim is
true inside the window it was measured and false outside it — the same shape as
the 5.80-of-8 probe number that was quoted as an engine result.

## Worse than an untuned constant: the wrong functional form

`verify_lens` maximizes `(R + total) / (bias + row·(R + i))` — cost affine and
**monotone** in the row count, with `BIAS_MS = 211.0` and `ROW_MS = 0.53`
imported from agent-infer's H20 numbers.

The measured cost on this backend, from the 08-30 entry's own table:

| rows | replay ms | marginal |
|---:|---:|---:|
| 1 | 10.76 | — |
| 2 | 17.54 | +6.78 |
| 3 | 27.06 | +9.52 |
| 4 | 27.58 | **+0.52** |
| 8 | 27.13 | **-0.11** |

`linear_*_mma8` pads M to 8 unconditionally, so rows 5 through 8 are free and
row 9 starts a new arm. The true cost is a **staircase with a flat tread and a
cliff**, and it is not monotone.

An affine model cannot represent that, which makes the failure worse than
over-eager trimming. **The policy can prefer a shorter chain that costs the
same or more**: trimming 7→6 or 6→5 inside the padded region gives up a draft
and saves nothing. The 227 trims at offered 7 sit in exactly that region.

So there is no correct scalar for `ROW_MS`. Re-measuring it would have looked
like progress and produced a model still unable to express the shape. A
borrowed constant can be re-measured; a borrowed functional form cannot be
fixed by re-measuring it — and the flat region is not staleness in agent-infer's
numbers, it is `_MX=8` padding in our fp4 arms, which their cost function has no
term for.

## Why the shipped path is safe, and only by accident

`--depth` defaults to 2 (W=3), and `build_engine` takes the head's own default,
which is 2 for the chain head. Both are inside the never-trims region. The
policy is **inert where we ship and wrong where it binds**, and nothing ships at
depth 6 or 7.

## Not fixed

Two designs would work and neither is a retune: carry the staircase, or
restrict the policy to trims that cross an arm boundary. Which one is better is
a goodput question, and goodput on the wide tick is not measured yet.

Deleting the policy is off the table — it is live at the depths it was about to
be deleted for not reaching.

## Rule

A measurement establishes a claim over the range it sampled. "Always returns
its maximum" was measured at depths 1, 2 and 4 and stated without them; the
policy's binding region starts at 6. Record the range beside the claim, the way
the MMLU entry now records the slice and the concurrency beside the score.

Before retuning a borrowed constant, check that the borrowed *shape* fits. A
constant with no correct value is telling you the model is wrong, not the
number.
