# The step is 74% rollout, and "backward_secs" was 20% forward — 27B, 2026-09-07

## Context

Two numbers priced every RL lever, and neither was what its name said.

`rl_step` timed forward + loss + backward as one bucket and called it
`backward_secs` (`train.py:337`, whose own comment admitted "includes the recorded
forward"). And the step denominator came from
[a16ff9c](2026-09-06-one-grpo-step-is-54-percent-backward.md) — 131.579 s, backward
54.4%, rollout 45.6% — a tree from before the chunk ladder landed.

This splits the bucket and re-measures the denominator on current main.

## What Worked

`forward_secs` is carved out at the seam that already existed: `_step.run()` closes its
`with torch.no_grad(), tape:` block, then `_sync(backend)` drains the device, then the
clock is read. Without that sync the forward's launches are still in flight and its GPU
time is billed to whichever phase synchronizes first — the backward.

`other_secs` is a derived remainder, and `invalidate_secs` times
`engine.invalidate_weights()`. `backward_secs` keeps its published meaning, because
`tests/test_recipes.py` sums it with rollout and optimizer to reconstruct the step;
`backward_only_secs` is the new, narrower number.

## The step, warm, on current main

`qwen38-27b`, gen 1024, group 8, micro 1, tp=1, card 0, at `73433bb`:

| phase | secs | % of step |
|---|---|---|
| rollout | 63.156 | **73.77%** |
| forward | 4.415 | 5.16% |
| backward-only | 17.957 | 20.97% |
| optimizer | 0.087 | 0.10% |
| invalidate | 0.001 | 0.00% |
| other | 0.002 | 0.00% |
| **step** | **85.617** | 100.00% |

The phases reconstruct the step to **0.00e+00 s**.

## The ruling this measurement was scheduled to confirm is now false

a16ff9c concluded "the next RL lever is backward, not the rollout". At the same shape on
current main:

| | a16ff9c | here | ratio |
|---|---|---|---|
| step | 131.58 s | 85.62 s | 0.651x |
| rollout | 59.97 s (45.6%) | 63.16 s (**73.8%**) | **1.053x** |
| backward bucket | 71.53 s (54.4%) | 22.37 s (**26.1%**) | **0.313x** |

**The rollout agrees to 5%.** That is the cross-check that makes the rest readable: the
rollout path has not changed, so the flip is not a measurement artifact.

**The backward got 3.2x faster**, which is expected rather than anomalous — a16ff9c
predates the chunk ladder. Merged since: `f0e6e71` chunk 16→64 at 1.94x (#229),
`88e3764` chunk 64→128 (#234), `8d6a24a` the fp4 backward running Ampere MMA on a Hopper
card (#240), `16e8cd0` the fp8 frozen dX at 1.042x (#263). The 1.94x alone plausibly
carries most of it.

So the conclusion inverts. **10% off the backward is now 2.6% of the step; 10% off the
rollout is 7.4%.** Any lever priced as "% of step" through the backward bucket needs
re-reading against 26.1%, not 54.4%.

### The GDN port's ceiling moved because the backward was optimized out from under it

the GDN backward port design prices the port as `backward_secs` 23.194 → 15.774,
**1.470x**, and calls that "on the step". It is a ratio on the **bucket** — the two were
close enough to conflate when the bucket was 54% of the step, and are not now.

Re-priced on the measured step: 1.470x on a 22.372 s bucket saves 7.153 s of 85.617, so
the ceiling is **1.091x on the step**. Against a16ff9c's 54.4% bucket the same 1.470x
would have read **1.211x**. Nothing about the port changed; the denominator did, because
#229/#234/#240/#263 optimized the backward while the port was being designed.

That ceiling is already stated in its own doc as an upper bound that will not be reached,
for four named reasons. 1.091x is therefore the optimistic end of a decision that now has
to clear a much lower bar — and the general lesson is the one that generalizes past this
port: **a lever priced against an old profile is priced against a denominator that no
longer exists.**


## A mislabelled denominator is a sign problem, not a precision problem

`backward_secs` is **19.7% forward** (4.415 of 22.372 s). A lever measured as a ratio *R*
on that bucket has a true ratio of `1/(1 - s/(1-F))` on the backward and
`1/(1 - s/F)` on the forward, where `s = 1 - 1/R`. Those go in **opposite directions**:

| R on bucket | backward-lever | forward-lever | relative correction |
|---|---|---|---|
| 1.043 | 1.0539x | 1.2688x | 1.26x / 6.25x |
| 1.042 | 1.0527x | 1.2613x | 1.25x / 6.22x |
| 1.020 | 1.0247x | 1.1128x | 1.24x / 5.64x |

So two levers priced against this bucket **are not comparable unless they touch the same
half** — which is what makes the name expensive rather than merely imprecise. (Framing
owed to `v100-sm70-fp4-55`.) The forward exposure is `1/F` = 5.08x of a reported gain.

Neither standing verdict flips: the GDN adjoint rejection is 1.043 → **1.054x**, still a
reject; the fp8 dX accept is 1.042 → **1.053x**, still an accept. Both are backward
levers, and the backward correction is the small one.

**F is stable across the two steps: warm 19.7%, cold 19.5%.** That answers the obvious
objection — if a forward-side kernel had JIT'd inside the timed forward window, cold F
would be inflated and warm F materially lower. They agree to 0.2 points.

## Two readings this run would produce in a careless reader

**Step 1's rollout was 660.1 s — 11.0x a16ff9c's warm 59.965 s.** That is the
decode-kernel first-compile, inside the timed rollout region, not a rollout regression.
Step 1 is 96.4% rollout for that reason alone, and only step 2 is a measurement.

**The pool was checked before the step, not after.** `room_for(prompt 256)` reported 7936
tokens against the 1024 asked, with 680 usable blocks, and the script exits 1 when they
disagree. `tilerl-48` spent five arms on 2026-09-07 producing numbers that were
self-consistent about a pool of the wrong size, because their probe reported the sizes and
continued. A guard that refuses beats a guard that annotates.

## Rule

A tolerance has to be sized against the quantity it protects, not against the total. The
first version of the reconstruction gate allowed 1% and the mutant that zeroed
`other_secs` entirely **passed** — the remainder is 0.054% of the tiny step, so the band
was 18x looser than the term it guarded. The identity is arithmetic, not a measurement, so
it is now exact.

The same error at a larger scale is what this entry corrects: `backward_secs` was a
bucket whose name asserted something its contents did not, and the 20% band in the old
gate was wide enough that nothing ever noticed.

## Results

| date | commit | machine | what | result |
|---|---|---|---|---|
| 2026-09-07 | 73433bb | H20 card 0 | warm step, gen 1024 | **85.617 s**, phases reconstruct to 0.00e+00 |
| 2026-09-07 | 73433bb | H20 card 0 | rollout | **63.156 s, 73.77%** |
| 2026-09-07 | 73433bb | H20 card 0 | forward | 4.415 s, 5.16% |
| 2026-09-07 | 73433bb | H20 card 0 | backward-only | 17.957 s, 20.97% |
| 2026-09-07 | 73433bb | H20 card 0 | `backward_secs` forward share | **19.7% warm, 19.5% cold** |
| 2026-09-07 | 73433bb | H20 card 0 | cold step 1 | 684.5 s, 96.4% rollout (first-compile) |
| 2026-09-07 | 73433bb | this Mac | full suite | 472 passed, 15 skipped, 1 xfailed |
| 2026-09-07 | 73433bb | this Mac | two mutants | both fail the reconstruction gate |

## Limitations

- **One warm step**, like a16ff9c. No spread.
- **73.8% is one point, not a curve.** gen 1024, group 8, tp=1, one warm sample. Longer
  generations only grow the rollout share, but nothing here says how much.
- **This measures where the time is, not whether it can be recovered.** 63.156 s for 1024
  ticks × group 8 may be near the decode hardware bound or may have slack, and this run
  cannot tell. A rollout track needs that headroom probe first; it is a separate window.
- **micro=1 with a LoRA adapter**, so this is not the full-parameter 50.1 GiB path.
- **tp=1.** TP=2's split is a different measurement; [#270](2026-09-07-tp2-on-two-cards.md)
  has the collective side of it.
- `invalidate_secs` reads 0.0008 s with `recapture_graph=True`. That is a real reading on
  this configuration, but `_const_f32`'s cached cast refills only when called and a replay
  calls nothing (noted by `v100-sm70-fp4-55`), so a configuration that replays more may
  pay elsewhere.

