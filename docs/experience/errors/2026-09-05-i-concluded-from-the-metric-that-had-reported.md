# I concluded from the metric that had reported, not the one the run optimizes

**Date:** 2026-09-05
**Run:** `8ca073e54686` — the 2048-cap GRPO control for
[the thinking cap](../wins/2026-09-04-the-thinking-cap.md).

## Context

The control finished its 100 training steps and printed, in this order:

    peak allocated 82.37 GiB
    adapter 170.8M params -> runs/8ca073e54686
    mmlu 0-shot 730/1000 = 73.0%          <- after
    ... 56 minutes of silence ...
    gsm8k greedy 482/500 = 96.4%          <- after

MMLU landed first because it runs at `max_new_tokens=1`. GSM8K runs 500
questions uncapped and took most of an hour.

In that hour I wrote the entry, opened the PR, and put this in both:

> an arm this starved should not be expected to move a downstream metric at
> all, and it did not.

## Root cause

The run is a GRPO run against a GSM8K reward. **GSM8K is the metric it
optimizes.** MMLU is the out-of-domain check. I had the out-of-domain number,
pointing down by 2.2 points, and wrote a conclusion about whether the run
learned anything — while the in-domain number had not printed.

Two things made it feel safe rather than premature:

1. **The number I had agreed with my model.** 92 of 100 steps were fully tied,
   so "starved run, no effect" was the shape I expected, and MMLU appeared to
   confirm it. A confirming reading gets less scrutiny than a surprising one.
2. **I had already hedged it correctly.** The entry says the 2.2 points are
   n=1 and not to be read as a regression — a peer had made exactly that point
   and I had agreed. So the sentence *looked* careful. It was careful about the
   number's precision and careless about which question the number answers.

## What the run actually did

| GSM8K, uncapped, n=500 | accuracy | vs base |
|---|---:|---:|
| base | 448/500 = 89.6% | — |
| 2048-cap arm | **482/500 = 96.4%** | **+6.8 pts, z=4.25, p=2.1e-05** |
| 256-cap arm | 474/500 = 94.8% | +5.2 pts |

MMLU 75.2% → 73.0% is z=-1.12, p=0.26 — not significant, exactly as hedged.

**Eight gradient steps moved GSM8K 6.8 points**, more than the arm that took
28 of them, and the two arms are indistinguishable from each other
(+1.6, z=1.23, p=0.22).

## The substantive correction

`tied_group_fraction` measures **efficiency, not efficacy**. A tied group
contributes a zero advantage and costs wall clock; it does **not** dilute the
steps that are not tied. Each of the 8 live steps came from a group where the
policy disagreed with itself, which is the condition under which a
REINFORCE-with-baseline step carries information.

The right complaint about this run is that it spent 5.5 hours to take 8 useful
steps. Not that it learned less. I had been treating a 92% tie rate as a
measure of how little the run learned, and it is a measure of what the learning
cost.

## Fix

Deleted the sentence, rewrote the entry around the real result, and corrected
the README section that had inherited the wrong mechanism from me hours earlier.

## Rule

**Do not conclude while the metric the run optimizes is still running.** Name
the primary metric before the run starts; every other number that lands first
is context, not an answer. When an early number agrees with your model, that is
the moment the conclusion is cheapest to write and hardest to retract — the
agreement is what removes the friction that would have made you wait.

A corollary for this repo, since the same shape recurred four times in two
days: **hedging a number's precision is not the same as checking that the
number answers the question.** "n=1, not significant, do not read as a
regression" is correct about MMLU and says nothing about whether MMLU was the
metric to read.
