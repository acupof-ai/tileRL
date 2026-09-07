# The backward's chunk, 64 -> 128: a further 1.194x, and where the ladder stops — H20 sm90, 2026-09-07

> Status: ACCEPT for 128. `_GDN_CHUNK = 64 -> 128`, `backward_secs` 41.446 -> 34.717 on the #192
> recipe. **256 measured and REFUSED**: 1.063x for 2.05 s, and the triangular solve's total rises.

## Context

[The C=64 entry](2026-09-07-c64-backward-chunk.md) established that both python loops in
`gdn_backward` are dispatch-bound: a chunk 4x larger cost the same per call, so the seconds
followed the call count and 16 -> 64 gave 1.935x. That predicts nothing about 128 — a prediction
would need the arithmetic to stay free, and the intra-chunk terms are quadratic in chunk length.
So the ladder was measured, not extrapolated.

## Four arms

Card 6, uncontended, sequential with the card verified empty between arms, the #192 recipe
(prompt 256, gen 1024, group 8), T=1280, `wy_kernels` forward, `mlp` segment, warm step,
`--inside-gdn`. Only `--gdn-chunk` differs.

| chunk | `backward_secs` | vs prev | attributed | unattributed | adjoint ms/call | recompute | solve |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 80.207 | — | 53.255 | 26.952 | 0.931 | 0.560 | 0.065 |
| 64 | 41.446 | **1.935x** | 14.525 | 26.922 | 0.955 | 0.556 | 0.083 |
| **128** | **34.717** | **1.194x** | 7.799 | 26.918 | 0.889 | 0.541 | 0.121 |
| 256 | 32.667 | 1.063x | 5.606 | 27.061 | 1.122 | 0.674 | 0.293 |

Checked, not asserted: `attributed + unattributed` reconciles to each arm's `backward_secs` within
0.001 s on all four rows, so a transcription error in either column would show up as a residual
that does not close.

## Why 128 ships and 256 does not

**The quadratic term turns at 256.** Through C=128 the per-call cost is flat or falling — the
adjoint 0.931 -> 0.955 -> 0.889 ms. At 256 every row turns up: adjoint 1.122 (1.26x), recompute
0.674 (1.25x), and the solve 0.293 (2.4x). The solve is the clearest case, because its **total
reverses**: 1.984 -> 0.640 -> 0.464 s across the first three arms, then **up** to 0.562 at 256 —
1.21x more time for half the calls. That is the arithmetic overtaking the dispatch saving, which
is the thing the C=64 entry said it could not predict.

**A fixed cost caps what is left.** The unattributed column is 26.952 / 26.922 / 26.918 / 27.061 —
flat to 0.5% across a 9.5x change in attributed work. It is the rest of the backward, which the
chunk does not touch (`--inside-gdn` times three `reference` functions, so every linear, rmsnorm,
attention and checkpoint lands there by construction; the registry arm attributes 30.789 s of it
at C=128). So `backward_secs` is asymptoting on ~27 s: 16 -> 64 captured 38.8 s of the 53.3 s
available, and everything past it divides the 14.5 s that remained. **256 buys 2.05 s and no
chunk value can ever buy more than ~5.6 s.**

Both reasons point the same way, and either alone would be enough.

## Precision, at the shipped T

Same estimator as `51e965e` (`gdn_backward` at chunk C against chunk 1, worst relative over all
eleven gradients, worst of seeds 0/1/2), run at T=1280 rather than the test fixture's T=128:

| chunk | chunks | worst rel | vs prev |
|---:|---:|---:|---:|
| 16 | 80 | 4.553e-06 | — |
| 64 | 20 | 7.478e-06 | 1.6x |
| 128 | 10 | 2.048e-05 | 2.7x |
| 256 | 5 | 2.787e-05 | 1.4x |

All four are inside the 1e-4 bar, so **precision does not decide this and the step time does.**
128 costs 2.7x the error of 64 and stays 4.9x inside the bar. The bar is provisional
([errors/2026-09-07-what-gradient-error-is-acceptable.md](../errors/2026-09-07-what-gradient-error-is-acceptable.md)),
which is why the error is reported alongside the ratio: if the bar moves to 1e-5, C=128 fails it
and C=64 does not.

The saturation the C=64 entry recorded is a **fixture artifact and absent here**: at T=128, chunk
128 and 256 are one chunk and give the identical number. At T=1280 all four are many chunks and
the ordering is clean. That is why this table was re-run at the shipped T instead of reusing the
gate's.

## The instrument, bounded

`--no-instrument` at C=128 measures 31.339 s warm (peak 63.8 GiB) against 34.717 instrumented, so
this arm's hooks cost 3.378 s and its 7.799 s attributed is an **upper bound**, not a measurement.
The registry arm's is bounded at 4.281 s the same way.

There is no per-call instrument cost across arms. Fitting `T = W + c·n` on the two instrumented
arms gives **-102.8 us/call**, because the arm with *more* timed calls (30720 vs 21936) is the
*faster* one — the model is refuted on those two points, not unresolved. The wrappers sit at
different depths, so the two call counts measure different things and their difference is not a
lever arm. Separating the residual 2.6% from run-to-run spread would need repeats, not a third
configuration; nothing here rests on it, so it was not run.

## A correction that reached the docs and not the code

The C=64 entry's residual paragraph was wrong and was fixed there. The probe that *prints* that
residual was not: `scripts/prof_backward_ops.py` had been set aside to `/tmp` while the C=64 branch
was rebuilt on a moved base, and the copy that came back predated the fix. It still carried the
retracted "the remainder is `Tape.backward`'s own loop" in three places — the `--no-instrument`
help text, the residual print, and the module docstring. Had this branch gone out unread, the tool
would have printed the claim its own entry retracts.

The fix is procedural and cheap: diff a restored file against the branch before staging it, not
just against what you remember editing. A file that leaves the working tree and comes back is a
file whose history you no longer share.

## Rule

Measure each rung. "Per-call cost was flat at 4x" is a fact about one step of the ladder, not a
law, and the step where it stops being true is the one that decides where to stop. The solve's
*total* rising while its call count fell is the signal — a per-call number alone would have read
as a 2.4x regression in a shrinking row.
