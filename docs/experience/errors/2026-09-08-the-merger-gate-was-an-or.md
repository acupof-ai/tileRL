# The P3 merger gate was an `or` where the roadmap says each — 2026-09-08

> Status: **fixed** — `tests/test_merge.py:55-80`, the conjunction the roadmap specifies.
> No runtime change: `iso_merge` is untouched and its numbers already satisfy the tighter
> gate. What changed is what the gate admits.

## Context

`--method iso` has been the CLI default since it landed (`cli.py:1232`,
`choices=["iso","average"], default="iso"`), so the ISO merger is on the production path.
The roadmap is explicit about which of its properties is the gate (`roadmap.md:120-123`):

> Self-merge and Σ₀ preservation are true by construction and are **smoke checks**; the gate
> is two tiny specialists **each** keeping their task better than plain averaging.

The test that implements it asserted:

```python
assert out["iso"][0] <= out["avg"][0] or out["iso"][1] <= out["avg"][1], out
```

An `or`. "Each" is a conjunction.

## What the `or` admits

Driving the real `iso_merge` on the tiny model, tasks A and B, with a merge that ignores
specialist B — which is what a K-collapse bug produces:

| arm | loss A | loss B |
|---|---:|---:|
| base | 22.335 | 21.990 |
| average | 18.460 | 17.550 |
| **iso** | **15.995** | **14.726** |
| **iso, specialist B dropped** | **15.660** | **21.954** |

The dropped-B merge is **0.036 below the base's own B loss** — it learned essentially
nothing about task B, and it is **4.4 worse than plain averaging** there. It passes the `or`
on the strength of its A number, which is *better* than the real merge's.

So the gate for a default-on mechanism could be satisfied by a merger that keeps one
specialist and discards the other, and would report that as ISO beating averaging.

The real merge wins on both tasks, so **tightening to `and` does not change today's
verdict** — it changes what a regression can hide behind.

## What the tightened gate catches, swept rather than assumed

#284's message says this loss comparison is what covers merge-math regressions, and names
two mutants that survive its own exact assertion. Both driven through the `and`:

| mutant | loss A | loss B | `or` | `and` |
|---|---:|---:|---|---|
| baseline | 15.995 | 14.726 | pass | pass |
| `ridge` 1e-3 → 1e-1 | 16.518 | 15.213 | pass | **pass** |
| `rho_keep` 0.9 → 0.1 | 21.646 | 21.110 | fail | fail |
| drop specialist B | 15.660 | 21.954 | **pass** | **fail** |
| return the base | 22.335 | 21.990 | fail | fail |

`ridge` surviving looks like a hole and is not. Swept across nine decades, with the merge's
own step size as the mechanism:

| ridge | loss A | loss B | mean step / norm | `and` |
|---:|---:|---:|---:|---|
| 0 | 15.991 | 14.722 | 0.0137 | pass |
| 1e-3 | 15.995 | 14.726 | 0.0137 | pass |
| 1e-1 | 16.518 | 15.213 | 0.0125 | pass |
| **1.0** | 19.106 | 17.974 | **0.0069** | **fail** |
| 1e2 | 22.319 | 21.973 | 0.0001 | fail |
| 1e4 | 22.335 | 21.990 | 0.0000 | fail |

`ridge` is monotone with a knee at 1.0. It is *relative* to Γ's scale (`merge.py:53`), so
1e-1 is still a working merge — 0.0125 of `|W|` of movement, beating averaging on both
tasks — and the gate fails from 1.0 up, where the merge stops moving the weights. **The
mutant that survived was inside the parameter's flat region, not outside the gate's
sensitivity.** #284's claim is right about the direction and wrong to imply a gap.

## Rule

**A gate quoting a spec must copy the spec's quantifier.** "Each of two" became "either of
two", and the difference is exactly a merger that drops half its inputs. The roadmap sentence
and the assertion were both in the tree, three lines apart in intent, for the whole life of a
default-on mechanism.

**A surviving mutant is a finding about the parameter before it is a finding about the gate.**
The reflex is to widen the gate until the mutant dies. Sweeping first showed `ridge` has a
knee, that 1e-1 sits below it, and that the gate's sensitivity is correct — so the right
response was to record the knee, not to tighten a threshold against a value that is not a
defect. Tightening would have made the gate fail on a working merge.

**A test that only compares against the base is weaker than it looks.** Both surviving
degenerate merges beat the base on at least one task; the base is a low bar. Averaging is the
comparison that has teeth, which is why the roadmap names it, and it only has teeth on both
tasks at once.
