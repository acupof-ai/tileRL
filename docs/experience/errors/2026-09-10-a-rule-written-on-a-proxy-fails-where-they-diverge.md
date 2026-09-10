# A rule written on a proxy fails where the proxy and the thing diverge

## Context

The 2026-09-10 P1 GRPO collapse had a measured mechanism: the length term ran
1.3x the correctness term at step 1 (2.1x+ after step 25), because a group
whose correctness is constant gets its gradient from length alone —
normalisation cancels the penalty's weight and the length spread fills the
whole advantage. The prescribed fix: zero a group's advantages when its
in-group correctness is constant (all correct or all wrong).

The rule as stated used **binary correctness constancy** as its criterion.
The reason behind it was "the only thing still varying in this group is
length". Those two are equal only in a world without the judge. Stage 4(b)
(a702c9a, judge verdicts become GRPO advantages) is exactly the world where
they diverge: in an all-pass group the binary correctness is constant, but the
judge scores vary — and the judge's ordering is the signal GRPO should learn
from. Applied literally, the rule zeroes judge-separated groups and stage
4(b) dies on arrival in the most common groups (all-pass is the majority at a
91% base).

## Root Cause

The rule was written on a **proxy** (correctness constancy) when the reason
was about the **thing** (no non-length signal). A proxy criterion fails
wherever the proxy and the thing stop moving together — here, the moment a
second signal source (the judge) enters the reward. The std gate has the same
blind spot in the other direction: a judge that cannot separate leaves the
judge term constant, but length keeps the reward spread non-zero, so "no
signal" looks like "signal" and the gate stays open.

## Fix

Criterion on the thing itself: `signal = reward − length_term`, where
`length_term = λ·tokens/cap` is a known deterministic function of the
completion (the inversion is exact, not estimated: `corr = reward +
0.1·tokens/512`, measured 2026-09-10). A group whose signal is constant over
its live rows yields zeros. One criterion covers all three cases:

| case | signal | result |
|---|---|---|
| no judge, correctness constant | constant | zeroed |
| judge on, judge separates | varies | kept (stage 4(b) intact) |
| judge on, judge cannot separate | constant | zeroed (the std gate's hole) |

The caller provides `signal` as a fact (judge scores when the judge is on,
reward minus the length term otherwise); the zeroing policy stays inside
`group_advantages`. Tests are a four-part shape: all-correct zeroed,
all-wrong zeroed, mixed group still judged with the right direction (the
control — without it, a bug that zeroes EVERY group is green), and a
judge-separated all-pass group keeps its ordering (the stage 4(b)
regression).

## The mechanism was specified, not missed

The old test pinned the collapse as correct behavior. This is
`tests/test_rl.py` before this fix:

```python
def test_all_wrong_group_ranks_advantages_by_length():
    """Every match is 0, so the length term is the whole reward: r_i = -lam*L_i/cap
    normalizes to -(L_i - Lbar)/std(L) -- shortest first, lam-independent for lam > 0,
    and silent at lam = 0."""
    ...
    assert np.all(np.diff(a[np.argsort(lengths)]) < 0), a   # shortest first, strictly
    assert np.allclose(a, adv_at(1.0))                       # lam cancels
```

The docstring IS the collapse mechanism, verbatim: an all-wrong group makes
the length term the whole reward, normalization makes it shortest-first, and
λ cancels. The author understood the mechanism completely, derived it
correctly, wrote it down — and asserted it as the behavior the system should
have.

This is not a missing test and not a weak assertion. It is a test that
correctly pinned a WRONG spec. Any fix makes it red, which reads as a
regression. That is how the collapse survived: not unwatched, but watched in
the wrong direction.

The second assertion is the more expensive one. `np.allclose(a, adv_at(1.0))`
says λ is not a dial: in a tied group, λ=0.1 and λ=1.0 produce identical
advantages, so turning the length penalty down could never have fixed this.
The answer to "would tuning λ help?" was written in the test's second line
the whole time.

Part of P1's −13.4 pt is that this spec was never questioned — no number
attached, just that it was never questioned.

The test is renamed `test_an_all_wrong_group_is_silent_not_length_ranked`:
the new name states the behavior we WANT, the old name stated the behavior
we OBSERVED.

## Three places, and they form a chain

The old spec was pinned in three tests, and the three pins form a causal
chain. No single step is an error; each was a correct derivation from the
state before it. Together they pushed the problem downstream.

**1. The harmful gradient was manufactured into the spec.**
`test_all_wrong_group_ranks_advantages_by_length` pinned the all-wrong
collapse as correct behavior (quoted above).

**2. The same mechanism became load-bearing by design.**
A second test pinned the all-RIGHT collapse — and went further, calling the
length term a FIX:

> "Two correct answers of any two lengths are indistinguishable under a
> correctness-only reward, so an all-right group ties at zero advantage...
> Run 2 collapsed that way at step 41 of 100."
> "the first instinct on reading this fix is to tune it: in an all-right
> group lambda CANCELS"

The length term was INTRODUCED to break all-right ties (run 2 collapsed in a
tied group at step 41). The gradient this fix removes was not an accident the
tests failed to catch — it was the design.

**3. The instrument that could have noticed was blinded, and a second
instrument was built around the blind spot.**
`test_tied_is_structurally_zero_at_positive_lam` documented that at λ>0 the
continuous rewards make `tied` structurally 0.0 even for an all-correct
group — the metric that would have shown groups silently going dead could
not see them. Its conclusion was not "remove the length term" but "the
validity gate must read `tied_correctness`": a second metric was added to
route around the blindness instead of curing it.

Manufacture the harmful gradient → make it load-bearing by design → blind
the metric that watches it → add a second metric that routes around the
blindness. Every step is locally defensible; the chain is the collapse
surviving review.

The second test also stayed GREEN after the fix, because it never passed
`signal=` and the default keeps the old behavior. A green test asserting
what production no longer produces is worse than a red one: the red one
raises its hand, the green one does not. The sweep criterion after
overturning a semantics is therefore not "which test went red" but "which
test asserts the semantics we just overturned" — found here by the
structural sweep (every caller of `group_advantages`), not by a name search.

What is kept and what is cut: the length term is a real dial in MIXED
groups, where correctness varies — that half stays, and the test's
mixed-group section still proves it. In tied groups it was the ONLY dial —
that half is cut. The tie itself: the judge breaks it when it can separate
the rollouts (stage 4b); without a judge it is not broken, because a group
with no correctness difference has no learnable ordering to learn.

## Downstream: `tied_correctness` changed meaning

The number did not change; the world it describes did. Before the zeroing,
`tied_correctness = 0.6` meant "60% of groups have constant correctness and
their advantages are decided by length" — those groups still contributed
gradient, of the harmful kind. After the zeroing, the same 0.6 means "60% of
groups contribute no gradient at all": the effective batch size is 40% of
what the rollout count says.

The P1 validity gate reads `tied_correctness` with a threshold of 0.5, set
for the old meaning. It must be re-derived for the new one — a 60%-silent
batch may be more alarming than the old reading, or more tolerable because
the remaining 40% is clean signal; the inherited threshold is not
automatically right. Not done in this PR.

## Rule

用代理写判据,判据就在代理和本体脱钩的地方失效。Writing a rule on a proxy
when the reason is about the thing itself: the rule holds exactly where they
agree and fails exactly where they diverge. When receiving a specification,
ask what its reason is and under what circumstances rule and reason would
part company — then write the criterion on the reason, not on its proxy.

And when reviewing tests, the question is not "is this assertion written
correctly?" but "do we want this behavior?". The former passes review; the
latter is the gate. A test with a correct derivation, precise assertions,
and the wrong direction passes every code review — and pins the wrong
spec harder than no test at all.

When a fix overturns a semantics, sweep for every test that asserts the old
one — the red one raises its hand, the green one does not. The second test
here was green after the fix and still pinned the collapse; only a sweep for
"asserts the semantics we just overturned" finds it.
