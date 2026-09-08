# A probe whose model ran to the cap, reported as the recipe's blocker

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55
**Status:** open — the pool-exhaustion-raises half is unfixed; see the last paragraph of Fix
and the row in [OPEN.md](../OPEN.md).

## Context

I reported to the session coordinating tonight's work that the 27B curve run could not start:

> `--max-new-tokens 256 --eval-max-new-tokens 2048 --group 8` (the recipe's exact values)
> → `RuntimeError: PagedKvPool exhausted: all 521 blocks in use`

That reading is real and reproducible. It is also a property of my probe, not of the recipe.

## Root cause

The probe builds `--model tiny`, a random-weight 2-layer model. It emits no `<|im_end|>`, so
`engine.py:1370` never fires and **every row runs to the full cap**. Blocks are allocated
incrementally as a sequence grows (`engine.py:995`, `while len(r.blocks) * BLOCK_TOKENS <=
r.seq_len - 1 + q`), not reserved from the cap — so the pool holds actual lengths, and the
probe's actual lengths were 2048 where the 27B's are 322.

| length | blocks/row | × 8 rows | against a 520-block pool |
|---|---:|---:|---|
| GSM8K mean, measured today | 21 | 168 | fits |
| p90 532 | 34 | 272 | fits |
| p99 / the 1.3% at cap | 64 | 512 | fits, by 8 blocks |
| **tiny, runs to the 2048 cap** | **128** | **1024** | **exhausts** |

A 6.4x difference in completion length between the model I measured and the model the claim
was about. The same probe that correctly tests the sizing *arithmetic* cannot say whether a
real run exhausts the pool, because the quantity that decides it — how long the policy
actually generates — is exactly what a random model gets wrong.

**The conclusion survives, and the corrected arithmetic is worse than the original claim, not
better.** Two terms were missing from the table above. The eval arm runs greedy
(`eval.py:147` forces `temperature=0`), so its lengths are the 09-04 greedy table's — mean
320.8, p90 527, **max 926** at n=40 — not my temperature-1.0 ones. And the pool grows on
`r.seq_len`, which is **prompt + completion**: GSM8K prompts are ~183 tokens, 12 blocks a row,
96 blocks across 8 rows before any completion exists.

| greedy completion | + prompt 183 | blocks/row | × 8 | against 520 |
|---|---|---:|---:|---|
| mean 320.8 | 504 | 32 | 256 | 50.8% spare |
| p90 527 | 710 | 45 | 360 | 30.8% spare |
| **max 926** | **1109** | **70** | **560** | **exhausts by 7.7%** |

926 is a measured maximum, not an extrapolation. The curve run scores 500 rows four times —
50x the sampling of the table that produced 926 — so rows at or past it are near-certain, and
8 concurrent rows is the binding condition rather than all 8 being long.

So the report's conclusion (the run cannot start on merged main) was right, and its stated
reason was wrong by a factor of 1.8 in the block count. A correct answer does not audit its
own derivation.

## Fix

The report was corrected in the same channel within minutes, twice: first the model
substitution, then the two missing terms above.

The durable part: **a probe that substitutes a model has substituted every quantity that
depends on the model.** Sizing arithmetic is model-independent and the tiny probe tests it
correctly. Whether a pool is large enough is model-dependent, and there the tiny model is not
a smaller version of the 27B — it is a model with no stopping behaviour at all, which is the
worst case rather than a scaled-down case.

The failure is specifically **not** "the probe is wrong". The probe is correct about the thing
it tests. The error is that the conclusion crossed out of the probe's population, and the two
have different repairs: a wrong probe is fixed by fixing the probe, an over-broad conclusion is
fixed by scoping the conclusion. A peer hit the first kind the same day (a rollout probe that
never rendered the chat template); both produce a conclusion that looks verified.

Concretely: when a probe's answer depends on generated length, either drive it with a real
policy or compute the answer from a measured length distribution — **and use the distribution
the code path under test actually samples from.** I had the greedy distribution, I had the
temperature-1.0 distribution, I produced the second one two hours earlier, and I used the
second where the eval arm samples the first.

**Also open, and larger than any of this:** pool exhaustion raises. `engine.py:996`'s
`alloc_block` is called inside the decode loop and its own comment says "exhaustion raises and
`step()` fails the batch" — so one long completion takes down the other 7 rows on that tick,
and a 90-minute run can lose everything at step 70 because a few rows ran long. Correct
behaviour is preemption or requeue; every cap-arithmetic fix only moves the threshold.

## Rule

Before reporting a probe's failure as the system's failure, name the quantity the failure
turns on and ask whether the probe's value for it is the system's value. Here it was
completion length, the probe's was 2048, the system's is 322, and the ratio is the whole
finding.

This is the third instance today of an instrument answering an adjacent question cleanly —
after a host SVD timing reported as a card timing, and a `.float().cpu()` whose device
assertion tested the result rather than the transient. The shape is always that the probe is
correct about something, which is why the number looks trustworthy.
