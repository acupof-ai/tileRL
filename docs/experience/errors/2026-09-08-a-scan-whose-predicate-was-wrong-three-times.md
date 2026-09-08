# The self-comparison shape is not the defect, and the scan's predicate was wrong three times — 2026-09-08

> Status: **closed for `tests/`, one open finding in the roadmap.** Asked to enumerate gates
> a no-op implementation would pass, on the hypothesis that "compares against its own
> starting point" is a defect class. Driven: every candidate in `tests/` kills its no-op. The
> hits are all in `docs/roadmap.md`, and one of them is an error bar off by √2 in the P1 spec.

## Context

Three gates were fixed or found weak today, and they looked like one class:

- `iso[0] <= avg[0] **or** iso[1] <= avg[1]` — a disjunction where the spec says each ([the
  merger entry](2026-09-08-the-merger-gate-was-an-or.md)).
- "average beats the base on both tasks" — true for an average that dropped a specialist.
- `gsm8k_after > gsm8k_before` — passes 48.7% of the time on noise ([the gate entry](2026-09-08-p1s-gates-cannot-see-p1s-target.md)).

The hypothesis: gates that compare a mechanism against its own starting point, with no
second implementation, are a class. Scan the tree for the shape.

## Three predicates, and the first two failed in opposite directions

**Predicate 1 — "the assertion compares against a fixed reference": 76 hits, almost all
wrong.** The matches were overwhelmingly *fixture-liveness guards*:

```
assert len(asked) >= 1, "no tick reached the graph path; the test proves nothing"
assert st["ssd_hits"] >= 1, f"no hit ({st['ssd_recovered']} recovered), so this proves nothing"
assert backend.refill_const_f32() >= 1, "the walk refilled nothing; every arm below is vacuous"
```

These are the **opposite** of the defect — they exist to stop a vacuous pass. A predicate
that cannot tell a guard against vacuity from a vacuous guard is not measuring the property.

**Predicate 2 — "the assertion text names a quality metric AND a baseline": 0 hits,
including both gates I had fixed hours earlier.** The reason is worth keeping:

```python
assert out["iso"][0] <= out["avg"][0], f"A regressed against averaging: {out}"
```

No word for the metric, no word for the baseline. `out` is built lines above; the arm names
are dict keys. **The property lives in the test, not in the assertion** — so a
line-level predicate cannot see it, and zero hits looked like a clean answer.

**Predicate 3 — the test computes a quality metric and asserts an ordering on it: 18
tests, both known positives included.** The unit had to be the function.

## Driven, not argued: every candidate kills its no-op

Four candidates had the target shape — a reference that is the run's own starting point,
no second implementation.

**`test_train_loss_decreases` and `test_adafactor_trains_with_factored_state`** (`last5 <
first5`, `test_e2e.py:1899`, `:1918`). Three no-ops, all fail:

| arm | first5 | last5 | gate |
|---|---:|---:|---|
| AdamW lr=3e-3, as written | 18.0973 | 0.8525 | pass |
| AdamW lr=0 | 23.4357 | 23.4357 | **fail** |
| Adafactor lr=1e-2, as written | 9.5322 | 0.0118 | pass |
| Adafactor lr=0 | 23.4357 | 23.4357 | **fail** |
| AdamW that computes its update and discards it | 23.4357 | 23.4357 | **fail** |

The third arm matters: `lr=0` is a legitimate configuration, so it does not model a defect.
An optimizer that runs every path and then restores the parameter is the shape of a missing
`p.copy_`, and it dies too. **A 17-point fall cannot be faked**, which is what makes the
self-comparison sound here.

**`test_grpo_loop_raises_reward`** (`last > first` over 3-step means, `test_rl.py:290`).
This one has no stated margin against a stochastic rollout, so the no-op's false-pass rate
*is* the gate's floor. Eight seeds each:

| arm | passes | mean Δ | range |
|---|---:|---:|---|
| lr=0.05, as written | **8/8** | +0.4606 | [+0.3704, +0.5556] |
| lr=0, the no-op | **0/8** | −0.0405 | [−0.0556, −0.0185] |

Not one seed of the no-op crosses zero, and the two ranges are half an order of magnitude
apart with no overlap. The gate has a margin; it just never wrote one down.

**`test_an_all_pass_group_gets_signal_from_the_judge_alone`** (`adv[0] > adv[1] > adv[2]`,
`test_judge.py:49`) already carries its own negative control on the next line —
`assert np.allclose(group_advantages([1.0, 1.0, 1.0], 3), 0.0)` — which is exactly the
no-op.

## What the three real defects actually had in common

Not self-comparison. **A gate whose reference arm can move.**

| defect | reference | how it moved |
|---|---|---|
| the merger `or` | plain averaging | the disjunction let one task's win stand for both |
| the average control | the base | a degenerate average still beat it, by 0.049 |
| `gsm8k_after > gsm8k_before` | the same model, re-evaluated | noise, at 48.7% |

In all three the reference was another *measurement* — of a second arm, or of the same
system twice — and nothing pinned how far apart the two had to land. The sound gates all
compare against a reference that **cannot move**: a starting loss on a fixed batch and seed,
a `[1.0, 1.0, 1.0]` literal, an analytic derivative.

So the class is not "self-comparison". It is **"the gap between the two arms is not stated"**
— and a self-comparison against a frozen starting point is one of the *safest* forms, because
the starting point is a constant.

## The other half of the scan, where the hits are

`tests/` was the wrong place to look. The 11 `Exit:` criteria in `docs/roadmap.md` are where
an unstated gap survives, because prose has no assertion to run. Four carry an unquantified
comparison word:

| criterion | word | what it resolves to in code |
|---|---|---|
| `roadmap.md:107` P3 optimizer | **SFT loss falls**, spectrum **preserved** | `losses[-1] < losses[0] - 0.1` (`test_iso.py:85`) — a real margin, written only in the test |
| `roadmap.md:119` per-step requant | **MMLU flat** | `mmlu_before - 0.03` (`cli.py:850`) |
| `roadmap.md:126` merger, pod | **MMLU flat** | same |
| `roadmap.md:174` sequence parallel | **MMLU flat** | same |

`MMLU flat` appears three times and is unquantified in all three, while the P1 spec 60 lines
up writes the quantity explicitly — `MMLU (1000 q) after ≥ before − 2 pt` — and the code
implements **−3 pt**. So one document holds two different numbers for the same gate and a
word for it in three other places. (`tilerl-0a` found the −2/−3 disagreement independently;
this adds that the word appears three more times with no number at all.)

**And the P1 spec's own error bar is the wrong one.** `roadmap.md:57` reads
`GSM8K held-out (500 q) after − before ≥ +5 pt (SE ≈ 2 pt)`. At n=500:

| quantity | p=0.40 | p=0.50 |
|---|---:|---:|
| single-arm SE | 2.19 pt | 2.24 pt |
| **SE of the difference** (what the gate tests) | **3.10 pt** | **3.16 pt** |

`SE ≈ 2 pt` is the SE of **one** evaluation. The gate compares two, so its noise is larger by
√2, and quoting the single-arm figure understates the spread on the compared quantity by 42%.
That is what makes +5 pt look like 2.5 sigma when it is 1.6 — and 1.6 sigma is below the
7.70 pt an 80%-power test needs ([the gate entry](2026-09-08-p1s-gates-cannot-see-p1s-target.md)).
The paired test fixes this by making the SE genuinely smaller rather than by mis-labelling it.

## Rule

**A scan's predicate has to be checked against a known positive before its output is
read.** Predicate 2 returned zero, which reads as "the tree is clean". It was wrong on the
two examples that motivated the scan, and both were in the tree while it ran. One known
positive would have caught it in seconds; I ran it on the whole tree first.

**A predicate that matches the defect's inverse is measuring something else.** 76 hits
dominated by anti-vacuity guards was the signal that "compares against a fixed reference"
is orthogonal to the property. The volume looked like a finding.

**The unit of a property is where the property lives.** These gates keep the metric in a
variable and the arms in dict keys, so no assertion-level regex can see them. Scanning
functions found both positives immediately.

**Report a null result with its negative controls, not as an absence.** "No instances
found" is worth nothing without evidence the instrument could have found one. Here the
instrument is shown working on 18 candidates, of which four had the shape and four killed
their no-ops.

**And the hypothesis was mine to falsify, not to confirm.** Three fixes in one evening
looked like a class; the class was real but its defining property was the *unstated gap*,
not the surface shape. Scanning for the surface shape would have produced a list of sound
gates to "fix" — which is the same error as tightening a threshold until a mutant dies,
recorded twice today already.
