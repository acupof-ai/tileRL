# The ISO guard was reading cuSOLVER's accuracy, not f32's

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55

## Context

The first ISO-RL arm on the real 27B reached `iso.py:74` and raised:

```
ValueError: ISO: frames in torch.float32 are not orthonormal (max|UᵀU−I| = 4.0e-03);
change precision.py, not this call
```

The message points at the dtype, and `precision.py:15` does set `"frame": torch.float32`. Both
readings — "f32 is too coarse for a 5120-wide SVD" and "raise the tolerance" — are wrong.

## Root cause

Measured per shape class on the 27B's masters, host against card, same dtype:

| shape | k | host f32 | card f32 | host f64 | card ≤ 1e-3 |
|---|---:|---:|---:|---:|---|
| (248320, 5120) | 5120 | 6.56e-06 | **4.22e-03** | 1.04e-14 | False |
| (5120, 17408) | 5120 | 6.02e-06 | **2.54e-03** | 1.06e-14 | False |
| (17408, 5120) | 5120 | 5.95e-06 | **2.31e-03** | 1.17e-14 | False |

**Same dtype, 400–700× apart.** 4.22e-03 on the embedding is the 4.0e-03 the arm reported.
Host f32 sits 152× *inside* the 1e-3 guard; card f32 misses it by 4.2×. So the guard is
reading cuSOLVER's accuracy at k=5120, not float32's.

`iso.py:69` computes `torch.linalg.svd(p.to(frame_dtype))` **on the device the param lives on**.
The `.cpu()` is at `:78` and applies to the finished frames. Params are on the card after
`materialize`, so the SVD is cuSOLVER's. And the frames are moved to the host anyway —
`_offloaded` is true for every CUDA param (`:82`) — so computing them on the card buys 5× speed
and no residency benefit at all, in exchange for 400× the orthonormality error.

**My own probe reproduced the wrong path first.** Its first version wrote `.detach().cpu()`
before the SVD and reported "f32 passes everywhere, worst 6.56e-06" — a clean table with 152×
margin, answering a question ISO never asks. What exposed it was the contradiction: the arm
said 4.0e-03 and my probe said 6.56e-06, 640× apart. A discrepancy that large is the
instrument, and this time the instrument was mine.

## Fix

Three options, and the choice is not a precision question:

1. **Frames on the host.** Accuracy is ample. Cost measured, not extrapolated: 28.37 s for
   `(248320,5120)`, 24.09 s × 72 for `(5120,17408)`, 6.73 s × 144 for `(17408,5120)` — those
   three classes alone are **2732 s = 45.5 min**, one-time since frames are cached. Ten classes
   exist; four were measured.
2. **f64 on the host.** 1e-14, but slower (48.83 s vs 28.37 s) and doubles a frame footprint
   that is already 226 GiB.
3. **Relax the guard.** Rejected. Its own comment says a frame dtype that cannot hold
   orthonormality trains on a drifting spectrum, and ISO's entire premise is a frozen Σ. A
   4.2e-3 orthonormality error contaminates exactly the thing being measured, so this option
   produces a number that says nothing about ISO.

Option 1 is right on the merits; whether a 45-minute fixed cost is worth paying is a
`time_to_score` question, since ISO's claim is 2.7× fewer *steps* and a fixed overhead has to be
amortized against that. Left for the owner of that budget rather than decided here.

Two numbers in the tree need re-deriving, not assuming mine: `precision.py:13` says fp32 frames
are 200 GiB and summing over the measured shapes gives **226** (1.13×); and the frame footprint
is not a blocker either way, since 226 GiB is host-resident against 1928 GiB of RAM.

## Rule

An error message names the site, not the cause. This one named the dtype and the dtype was
fine; what differed was the *backend* running the same operation at the same precision. When a
guard fires on a numerical property, check where the number was computed before changing what
it was computed in.

And when a probe's reading disagrees with the failing run by orders of magnitude, the probe is
the first suspect — including when it is the newer, more careful measurement. Mine differed by
640× and was wrong.
