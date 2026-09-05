# The gate guarding this number cannot see this number — 2026-09-05

**Date:** 2026-09-05
**Arch:** target-independent (`_GDN_CHUNK` is on every target's backward path)
**Task:** looking for a lever in the GRPO step's 28% torch-eager share

## Context

`reference._GDN_CHUNK = 16` carries a comment saying 16 rather than upstream's 64
is a precision choice, with figures. Chasing the step's torch-eager share I
measured chunk 64 at **2.171x** the wall clock of 16 (t=1024, CPU) and read the
comment as stale. It is not stale — I misread its denominator, and that is a
separate error recorded below. What survived the retraction is this:

**Every shipped gate over `gdn_backward` passes at every value of `_GDN_CHUNK`.**

## Measured

Worst relative error against the serial scan (chunk 1), worst over seeds 0/1/2,
b=1 t=128 nkh=2 nvh=4 dk=dv=16:

| `_GDN_CHUNK` | worst rel err |
|---:|---|
| 16 | 3.26e-06 |
| 32 | 5.46e-06 |
| 64 | 1.13e-05 |
| 128 | 1.25e-05 |

Against the tolerances actually enforced:

| gate | tolerance | chunk 64 sits |
|---|---|---|
| `test_gdn_bwd_spans_chunks` | 5e-5 | **18x inside** |
| `_autograd_gradcheck` default | 1e-2 | **3704x inside** |
| `_finite_diff_gradcheck` | 5e-2 | further still |

Confirmed by running them: with `_GDN_CHUNK = 64`, **8 passed, 2 skipped** across
every GDN test in `test_ops_parity.py`. Raising the constant 4x — a real
precision regression the codebase deliberately declined — is invisible to the
suite.

The decision that set the constant was priced by a **three-tier probe** built for
`51e965e`, comparing 16/32/64 against autograd on the serial forward across 3
shapes × 3 seeds. That probe never entered the suite. So the value is protected
by a measurement that exists only in a commit message and an entry.

## Fix

`test_gdn_backward_precision_tracks_the_chunk_size` in `tests/test_ops_parity.py`.
It asserts the **ordering** (16 more accurate than 64) rather than an absolute
bound, because the absolute number is a property of the machine's f32 reduction
order while "coarser chunks round worse" is the property the constant was chosen
on. Three seeds, because at one seed 64 measured *below* 32 and read like a
reversal of the tradeoff — it was noise, and that noise is what sent me looking
for a stale comment in the first place.

Negative control both directions: `_GDN_CHUNK = 64` **FAILED**, `= 32` **FAILED**,
`= 16` passes, while every pre-existing GDN gate stays green at 64.

## The two wrong explanations

Worth recording because both were plausible and both were mine or a peer's:

- **"The comment is stale."** It says chunk 64's worst relative error is *5-10x*,
  and I read that as 5-10x chunk 16's. Its denominator is the **serial scan**:
  16 is 1.3-2.2x serial, 64 is 5-10x serial, so the ratio between them is ~2.7x —
  exactly the 2.70e-6 / 9.88e-7 I had measured. My numbers agreed with the comment
  I was using them to doubt.
- **"Backward accumulates state across chunk boundaries, so fewer chunks means
  less error"** (a peer's, offered to explain why 64 measured no worse than 32).
  The data runs the other way: 64 is 3.5x worse than 16, monotone across seeds.
  A wrong mechanism that explains an anomaly is harder to dislodge than an
  obviously wrong one, because the anomaly vouches for it.

One thing did change since `51e965e` and it strengthens the original choice
rather than weakening it: the tape backward was **62%** of the step then and is
**33.74%** now, so the wall-clock prize for coarsening has shrunk while the
precision cost has not.

## Rule

**A gate whose tolerance is orders of magnitude from the value it protects is not
protecting it, and its green is not evidence about that value.** Before treating
a passing suite as permission to change a constant, find the measurement that set
the constant and check whether anything in the suite would move with it.

The structural sibling, from the same day on the V100 (`v100-sm70-fp4-55`):
`/health` reported `blocks_total: 256`, which is simultaneously the KV-fit
ceiling, the `--max-ctx 4096` cap, and the signature of a deploy that never took
effect — three mutually exclusive conditions with one number. **A reading that
confirms three incompatible things confirms none of them.** Different mechanism,
same failure: the instrument's resolution does not reach the distinction being
made.
