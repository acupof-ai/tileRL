# A contamination correlated with the treatment is bias, not noise — 2026-09-08

**Status:** the mechanism behind one measurement's sign flip, written as its own entry because
the arithmetic generalises past B=16. The measurements are in
[the B=16 entry](../wins/2026-09-08-b16-fits-and-three-predictions-were-wrong.md).

## The numbers

Two arms of one probe, B=8 and B=16 on a 27B, gen 1024. First run, no compile accounting:

| | B=8 | B=16 | verdict |
|---|---:|---:|---|
| compiles | 42 | **141** | |
| compile seconds | 142 | **237** | |
| wall clock | 217.5 | 507.9 | |
| **ms/token** | **26.55** | **31.00** | B=16 is **1.168x worse** |

Re-run with a gate asserting zero compiles inside the timed steps:

| | B=8 | B=16 | verdict |
|---|---:|---:|---|
| **ms/token** | **10.409** | **7.930** | B=16 is **0.762x — 23.8% better** |

**The conclusion did not get noisier. It reversed.** And the clean B=8 figure is 2.55x below
the contaminated one, so the contamination was not a rounding-scale effect in either arm.

## Why subtracting the compile time cannot fix it

The obvious repair is to measure the compile seconds and subtract them. It fails, and the
reason is the point of this entry.

**The contamination is correlated with the treatment.** B=16 compiled 3.4x as many kernels
*because it is B=16*: each new decode width recompiles `paged_attention_decode`,
`gdn_decode_fused` and `write_tokens`, and the wider arm reaches more widths. So the error term
is not an independent draw added to each arm — it is a function of the variable under test,
pointing the same way every time.

An independent contamination widens an interval. One that moves with the treatment shifts the
estimate. **Subtraction removes a magnitude; it cannot remove a correlation.** Even a perfectly
measured 237 s and 142 s, subtracted exactly, would leave the two arms measured over different
effective work, because compilation and execution interleave in one wall clock — the CPU-side
compile overlaps device idle in a pattern nobody accounted for.

## Why the usual warm-up convention is not a defence either

`prof_grpo_step.py` already discarded step 0 and averaged the rest, with a comment saying
"step 0 pays every JIT". That is a hope, not a property.

Which widths a run reaches depends on the rollout's length distribution, which is runtime data.
Reading the dispatch (`backend.py:689` gates the per-M GEMV path on `2 <= M <= _MGEMV` with
`_MGEMV = 3`, and `:697` keys the cache on `M`, while `:782` pads to `_MX = 8` so M=4..8 share
one entry) shows the compiling widths are M ∈ {1, 2, 3} — the *last three* rows of a shrinking
batch. `tilerl-48` observed that this narrowness makes the trigger sparser, and sparser is
more fragile, not safer: reaching M=3 and M=2 requires a specific pattern of rollouts finishing
at different times, and a step whose rows finish together skips them.

Here the convention happened to suffice — B=16's step 1 compiled 40, steps 2–5 compiled 0 —
and it sufficed *structurally* only because these arms have no `stop_token_ids`, so every row
runs to the cap and the shrink sequence is identical every step. A probe using the factory
sampler loses exactly that property.

**So three treatments of the numerator are one class, not three:** doing nothing, subtracting
an estimated compile time, and discarding a head step. Only a count of zero inside the timed
window is a statement that does not depend on where the compiles landed.

## The gate

`len(backend._kernels)` differenced across each step. `Backend._kernel` keys its cache on
`(name, factory args, kwargs)`, so one compile is exactly one new key — a count, not an
estimate. Nonzero in any timed step and the probe prints `REFUSED` and returns 1 rather than
reporting seconds.

It has to refuse rather than warn, for the same reason as the `--group` guard in the same
change: the failure produces a *wrong conclusion*, not a slow one, and a warning is read by
whoever is looking.

## Rules

- **A contamination correlated with the treatment is bias, not noise.** It shifts the estimate
  instead of widening it, so it can reverse a sign — and its direction is predictable from the
  correlation, which means "the effect was probably smaller" is not a safe reading either.
- **Subtracting the contaminant's magnitude does not remove its correlation.** Establish that
  it is absent; do not price it and deduct it.
- **Prove a measurement is clean with a count, not a convention.** Dropping the first
  iteration, subtracting an estimate, and doing nothing are the same class of unclean, and the
  first two look processed.
- **Ask whether the nuisance term is a function of the variable under test.** That question,
  asked before the run, is what separates a wide error bar from a wrong answer.
