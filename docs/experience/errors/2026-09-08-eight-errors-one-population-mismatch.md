# Eight errors in one day, five of them the same one: the population was not the one the claim was about — 2026-09-08

**Status:** all eight caught, each written up separately. This entry exists because writing
them separately hid what they have in common. Shapes only; each row links its own entry for
the detail.

## The table

Every row is a correct procedure run against the wrong set of things. What differs is which
part of the comparison carried the mismatch.

| # | what happened | mismatched part |
|---|---|---|
| 1 | a probe fed the model bare documents where the shipped path renders a chat template | **format** |
| 2 | a probe's `tiny` model emits no stop token, so every row ran to the cap — reported as the recipe's blocker | **population** |
| 3 | MATH's 229.2 s/step quoted against a GSM8K schedule | **population** |
| 4 | GSM8K's 95.3% (train split, n=32) used as the test-split base | **population** |
| 5 | tied fractions from two different rollout caps compared directly | **condition** |
| 6 | tied fractions at two different `--length-penalty` values compared directly | **condition** |
| 7 | two arms at equal *steps* consumed 20 vs 10 problems | **control** |
| 8 | `f = 0.3` fitted to an earlier run's curve, applied to this one | **population** |

**Five of the eight are the same error.** Not eight lessons — one lesson with eight
instances, and the instances look unrelated because the surface differs every time: a
tokenizer call, a model size, a dataset name, a data split, a flag value, an arm length, a
fitted constant.

## Why it repeats

A number carries its value and not its population. `229.2` is `229.2` whether it came from
MATH or GSM8K; `95.3%` does not say which split; `0.34` does not say which cap. Every one of
these numbers was correct where it was produced, and every one was read somewhere its
population did not hold. **Nothing in the arithmetic can fail**, so no check downstream of
the number can catch it. The check has to happen at the moment two numbers are placed side
by side, and that is the moment that feels like it needs no checking.

## Row 8 is the one that cannot be caught by looking for a mistake

Rows 1–7 each contain an operation that is wrong on inspection: the wrong function was
called, the wrong file was read, the wrong flag was set. Row 8 contains none.

`f = 0.3` was read off a real curve, from a real run, with a real derivation. Its provenance
is not merely valid — it is *better documented* than most parameters in the tree. The single
defective step is that the curve it fitted is not the curve it was applied to, and that step
leaves no residue in the parameter, in its derivation, or in the code that uses it. It
survived until a measured SE existed and put it 0.8 pt below the resolvability floor.

**An analogy-sourced parameter and a constraint-derived one are indistinguishable on
inspection, and the analogy tends to look the better grounded of the two, because it cites
data.**

tilerl-27's framing, which is what makes this worth one entry rather than eight: the other
seven each point at a wrong operation; this one points at nothing wrong except the
population.

## Rule

**Before placing two numbers in one sentence, name the population each came from.** If the
answers differ, the sentence is an assertion about their comparability, not an observation.
This is the whole rule; rows 1–8 are one violation of it.

**Label a number with its population where it is written, not where it is challenged.** A
figure recorded as "229.2 s/step" invites the error; "229.2 s/step, MATH L5, cap 2048"
cannot be misread the same way. The cost is a few words at write time against a re-derivation
at read time.

**A parameter's provenance can be entirely valid and still wrong.** "Where did this number
come from" is the wrong question — it has a good answer here. Ask: fitted on what, and is
that the thing I am applying it to.
