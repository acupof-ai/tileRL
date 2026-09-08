# The validity gate cost 9.47 GiB that tiny could never have shown

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55

## Context

`scripts/steps_to_reward.py`'s Σ-drift gate answers one yes/no question: does a free
optimizer move the base model's singular spectrum? If it does, the ISO 2.7× claim's premise
does not hold in this setting and no step ratio here carries it.

It answers that by taking `torch.linalg.svdvals(v.detach().double())` of **every** 2D
parameter. On the 27B, first arm, before step 1:

```
torch.OutOfMemoryError: Tried to allocate 9.47 GiB.
  GPU 0: 95.22 GiB total, 3.56 GiB free, 91.65 GiB in use by the model.
```

9.47 GiB is the embedding — 248320 × 5120 × 8 bytes — asked for in float64 next to a model
that already holds 91.65 of 95.22 GiB.

## Root cause

**The instrument's precision exceeds the question by orders of magnitude, and the excess is
what makes it unrunnable.** The verdict is a threshold on a percentage ("did Σ move more than
5%"). float64 SVD of every matrix resolves the spectrum to machine precision. Two doublings
were paid for nothing: f64 over f32, and every matrix over a fixed sample.

**And the failure cannot occur at the scale it was validated at.** tiny's largest 2D parameter
is 96,896 elements; float64 SVD of it is free. Every check on this script — the parity of the
drift function, the three-state verdict, the mutation testing, the negative controls — ran on
tiny and all of them passed. The defect is not in any of the logic they tested. It is in the
*resource cost at the real shape*, which a small fixture holds constant at approximately zero.

That is the general shape: a small fixture is chosen so the logic is exercisable, and it
thereby removes every cost that scales. Coverage of the logic is not coverage of the run.

## Fix

Measured before redesigning, because a sampled f32 sweep is only worth building if svdvals is
affordable at all — a gate that adds an hour per arm is not a gate. `scripts/probe_svd_cost.py`
prints the 2D inventory by shape class and times one svdvals per class on the card.

The redesign follows the cheapest form that still answers the question: a **fixed** sample of
matrices, f32, identical across arms. Fixed and identical matters more than which matrices —
two arms sampled differently produce drifts that cannot be compared, which would break the
gate in a way that still prints a number.

## Rule

Before an instrument runs at the real scale, price it at the real scale. Ask what precision
the verdict needs — a threshold on a percentage needs f32 and a sample, not f64 and a census —
and check the instrument's own footprint against the memory the thing being measured already
holds.

A validation suite on a small fixture proves the logic and says nothing about cost. When the
only difference between the fixture and the target is size, assume every size-dependent
failure is untested, and enumerate them before the run rather than discovering them at
`rc=1`.
