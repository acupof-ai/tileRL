# A gate too strict to be met — P1's paired test, 2026-09-08

**Status:** fixed. The threshold half is [a threshold in the wrong
units](2026-09-08-a-threshold-in-the-wrong-units.md); this is the other half of the same gate.

## Context

That entry corrected P1's exit thresholds to match `docs/roadmap.md:57-58`: GSM8K after − before
**≥ +5 pt**, MMLU **≥ before − 2 pt**. Correct encoding, and still not a usable gate — because
the test the +5 pt is judged by cannot resolve +5 pt.

## Root cause

`docs/roadmap.md` quotes `SE ≈ 2 pt`. Computed at `eval_n=500`:

| reading | SE | 80%-power one-sided MDE |
|---|---|---|
| unpaired, p=0.36 | 3.04 pt | 7.55 pt |
| unpaired, p=0.50 | 3.16 pt | 7.86 pt |
| unpaired, p=0.60 | 3.10 pt | **7.70 pt** |
| paired, 10% discordant | 1.41 pt | 2.77 pt |
| paired, 5% discordant | 1.00 pt | 1.96 pt |

**The roadmap's ≈2 pt is neither.** It cites an unpaired magnitude and takes a value below it.
Under the unpaired reading the minimum detectable effect at 80% power is **7.70 pt against a
+5 pt target**, so a real +5 pt improvement fails to register about half the time. An exit
gate's expensive failure is a **miss**, not a false pass: P1 spends a pod run and records "did
not pass" while the effect sat above the standard.

**The eval is already paired and the record already holds what a paired test needs.** `cli.py`
scores one `eval_rows` list in both arms (`:503`) and `gsm8k_accuracy` forces temperature 0, so
question *i* is the same question on both sides; `_write_eval_rows` writes one row per problem
with a `correct` field. `_write_eval_rows`'s own docstring records what was lost: *"P1 fell back
to the unpaired interval because only totals were kept."* The rows exist now — nothing read them.

## Fix

`_mcnemar` (`cli.py:376`) computes `b`, `c`, `delta = (c−b)/n`, `se = sqrt(b+c)/n`, `z`, and
`_paired_delta` (`:411`) reads the two arms off disk so it also works on the cache-hit path.
`_finish` records the result as `metrics["gsm8k_paired"]` **beside** the threshold, never
replacing it: the threshold is the roadmap's criterion and stays the gate; the paired test says
whether the observed move is resolvable, which the threshold cannot. Absent rows fall back to
threshold-only — which is exactly what P1 did once already.

`b + c == 0` returns a result with `se`/`z` of `None`, not a failure: the arms agreed on every
question, so there is nothing for a paired test to resolve. Unpairable input (mismatched
question sets, missing `i`, empty) returns `None`. Conflating those two would let a broken
pairing read as perfect agreement.

### Verdict gates and validity gates

`gates_pass` was `all(...)` over a flat list, so a **validity** gate passing contributed to
"P1 passed". Each gate now carries `kind`:

```
verdict  : gsm8k_improves, mmlu_holds          -- did P1 pass
validity : groups_untied, reward_rises, ce_falls, rollouts_within_cap  -- is the run interpretable
```

`reward_rises` is why this matters. Reward is the quantity GRPO optimizes, so it rising is the
definition of the optimizer working — not evidence for P1's claim that RL moves a *downstream*
number — and rising reward with a falling eval is what reward hacking looks like. It must never
be able to make P1 read `pass`. Reward *not* rising is informative (run 2 collapsed that way),
and that failure is coarse enough for a zero threshold, so this needs no invented number. That
is why `reward_rises` is classified here rather than given a threshold: `docs/roadmap.md`'s P1
criteria contain no reward term, so a number would be a standard no document authorises.

**The roadmap already draws this line**: the tied-group criterion reads *"< 50% (else the task
is too easy for this model and the run says nothing)"*. Saying nothing is not failing. So this
encodes a distinction the authority already makes — the same footing as the threshold fix.

The classification is **settled**, not pending: tilerl-27 ruled it with the derivation above.
Overturning it is a one-line change to `_VALIDITY_GATES`, which is why no open row is carried
for a decision that has been made. The threshold stays a bare `>` because reward *not* rising
is the run-2 collapse signature and that failure is coarse, and because `roadmap.md:57-58` has
no reward term to derive a number from.

`gates_pass` is deliberately **unchanged**: it is the process exit code, and an uninterpretable
run is not a success either. `verdict_of(m, kind)` reads one class and returns `None` when that
class has no scored gates — a third state the caller must not collapse, since a run whose
verdict gates were all skipped has not failed P1, it has not tested P1. `format_run` annotates
only the disagreeing case (`FAIL` with the verdict gates passing → `novalid`), so every existing
reader of field 3 keeps working.

### The cache's payoff, finally recorded

`metrics["eval_{tag}_secs"]` times both paths. `wins/2026-09-05-before-eval-cache.md` has been
`pending-remote` since it landed, and the mechanism is 55 lines in `cli.py` plus 129 of test —
worth it at 15 minutes per hit, not at 40 seconds. The number now falls out of P1's own run
instead of depending on someone remembering to time it.

**The first version of this cached the duration**, and the existing test caught it. The payload
is built as `{k: v for k, v in metrics.items() if "_before" in k}`, and `eval_before_secs`
matches that filter — so writing the metric before the cache write put a *duration* into a
result cache, and the hit path's `metrics.update(saved["metrics"])` then replayed it. Measured:
the miss paid 0.7447 s, the hit paid 0.0013 s, and the manifest recorded 0.7447 s for both. The
one number this change exists to produce would have been the only wrong one, and it would have
read as the cache saving nothing.

The elapsed time is now read **before** the write (so a hit's cost excludes the write only a
miss pays, keeping the two comparable) and stored **after** it. The test asserts both halves:
no `_secs` key in the payload, and the hit's recorded time strictly below the miss's.

## Verification

Six mutants, `__pycache__` cleared between each — an equal-length edit in the same second is
invisible to `.pyc` invalidation and reruns the previous bytecode (that failure voided the first
mutation run on the threshold half):

| mutant | test |
|---|---|
| `se` denominator `n` → `n-1` | **FAILS** |
| drop `_mcnemar`'s pairing check | **FAILS** |
| every gate `kind: verdict` | **FAILS** |
| `reward_rises` moved to verdict | **FAILS** |
| `verdict_of`'s `if scored else None` dropped | **FAILS** (after the case below) |
| record `eval_{tag}_secs` before the cache write | **FAILS** |
| control | passes |

The last one **survived** the first round: no case distinguished `None` from a pass, because
`all([])` is True. The added case uses `steps == 0`, which is the real path that skips every
gate — an *absent* metric is not a skip, it vacuous-passes by design, so the skip has to come
from the run's shape. A `None` collapsed to True would report a smoke-test invocation as having
passed P1.

## Rule

**A threshold and the test that judges it are one design, not two.** Encoding a criterion
correctly is not enough — check that the instrument can resolve the effect the criterion asks
for, at the n it runs at. And when quoting a noise figure, say which it is: a significance floor
(~5 pt here) and an 80%-power MDE (~7.7 pt) differ by 50% and imply different verdicts.

**A gate that measures the optimizer's own objective cannot be evidence for a downstream
claim.** Separate "did it pass" from "is the run interpretable", and let the authority's own
wording decide which is which — `docs/roadmap.md`'s "the run says nothing" was already the
distinction, unencoded.

**A duration must not travel through a result cache.** A result cache replays a past run's
values; a cost belongs to the run that paid it. The bug here needed no new code to appear — a
pre-existing `if "_before" in k` filter swept the new key in the moment it was written one line
too early. So when adding a metric, check every filter that already selects metrics by name
pattern, and put timings on the far side of the write.
