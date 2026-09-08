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

That probe reported **1698 s for a full f32 census** — 20 shape classes over 1540 2D params,
worst class `(17408, 5120) × 144 = 361.6 s`. It also printed `card: 0.0 GiB allocated`
immediately after `_build_model`, which is the tell I initially read past: **`load_hf` returns
host tensors**, so `v.detach().float()` stayed on the host and the `torch.cuda.synchronize()`
around it timed nothing. The 1698 s prices the **host** path.

`scripts/probe_svd_device.py` separates them by measurement instead of by argument:

| shape | n | host s | card s | ratio |
|---|---:|---:|---:|---:|
| (248320, 5120) | 3 | 10.62 | 4.33 | 2.45 |
| (5120, 17408) | 72 | 2.54 | 2.35 | 1.08 |
| (17408, 5120) | 144 | 2.38 | 2.37 | 1.01 |
| (12288, 5120) | 32 | 2.30 | 2.33 | 0.99 |

The magnitude survives (1.01–1.08× on the large classes — this SVD is algorithm-bound, not
bandwidth-bound), so sampling stays forced. **But the correction changes the fix.** In the
real run `build_engine` → `materialize` moves the params to the card before the gate reads
them, leaving 3.56 GiB free, and **one f32 embedding is 4.74 GiB**. A per-shape-class sample
necessarily includes `(248320, 5120)`. So f32 + sampling — the shipped fix — would still have
OOMed. The gate has to compute on the **host**, which costs ~6% of an already-sampled sweep
and zero card memory.

The final shape: a **fixed** sample, one member per shape class, f32, `.cpu()`. Fixed and
identical across arms matters more than which matrices — two arms sampled differently produce
drifts that cannot be compared, which would break the gate in a way that still prints a
number. `--sigma-per-class` defaults to **2, not 1**, because with one member per class the
in-run sample audit below can never fire: a default value can make a correct check
structurally unreachable.

Whether a sample carries the census verdict **cannot be settled on tiny**. Its census argmax is
`layers.0.o_proj` at all 8 steps of a real trajectory, and that is the sole member of its shape
class — no sample can exclude it. A naive ratio check read 1.000 at every step and an
adversarial non-argmax sample was equally unable to fail. So the audit ships inside the run:
the free arm reports the widest within-class drift spread, on classes with 3–144 members.

One assert in the self-check is **vacuous on a CPU host and says so in its own output**: it
asserts `spectra` returns host tensors, and `.float()` and `.float().cpu()` are both `cpu`
locally, so it can only fail on a card.

## Rule

Before an instrument runs at the real scale, price it at the real scale. Ask what precision
the verdict needs — a threshold on a percentage needs f32 and a sample, not f64 and a census —
and check the instrument's own footprint against the memory the thing being measured already
holds.

A validation suite on a small fixture proves the logic and says nothing about cost. When the
only difference between the fixture and the target is size, assume every size-dependent
failure is untested, and enumerate them before the run rather than discovering them at
`rc=1`.

And a timing needs its device read, not assumed. A `torch.cuda.synchronize()` around a host
tensor is a no-op that makes a host measurement look like a card measurement, and the tell was
printed in the probe's own header (`card: 0.0 GiB allocated`) two lines above the table I
quoted. The correction mattered: it did not move the magnitude, but it moved where the
computation has to happen, and the fix I had already shipped would still have OOMed.
