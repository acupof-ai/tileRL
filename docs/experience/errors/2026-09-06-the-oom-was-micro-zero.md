# The OOM was `micro=0`, not the buckets — 2026-09-06

> Status: open on two fixes. The cause is measured and needs no more card time;
> `--micro`'s help does not warn that 0 at group 8 / cap 2048 exceeds one H20, and
> `pod_sync.sh` still wipes every peer's `runs/`. Both listed in OPEN.md.

## Context

The [H20 snapshot](../wins/2026-09-06-h20-snapshot-at-faae3c8.md)'s GRPO run died
at step 2 with CUDA OOM at cap 2048, group 8, one H20, and I wrote that as a
capacity fact: "this configuration does not survive past step 1."

Review refuted it with a run on file: MATH run 2 (`0f7006c74ea0`) did **45 steps**
at group 8 and cap 2048 on one H20
([the rollouts grew into the cap](2026-09-06-the-rollouts-grew-into-the-cap.md)).
The proposed cause was growth between run 2's commit and `faae3c8`, with #140's
per-step bucket widths as the mechanism-level suspect: run 2 padded to one width so
the caching allocator reused blocks, buckets vary the width, torch keeps blocks per
size, and step 2 fragments.

## Root Cause

**The two runs differ on `micro`, which neither side had listed.**
`recipes.py:36` gives `grpo-math-27b` **`micro=1`**; my runs took the CLI default
**`micro=0`**. `train.py:137` is `rows = micro if 0 < micro < b else b`, so micro=0
puts all 8 group rows through **one** backward and micro=1 puts **one row at a
time** — an 8x difference in live activations at exactly the step that OOMs.

Probe: `--steps 3 --micro 1`, everything else identical to the run that died (same
data, cap, card, warm tilelang cache).

| | micro=0 (default) | micro=1 (run 2's config) |
|---|---|---|
| step 1 | 25.3 s, backward 8.070 | 39.6 s, backward 22.228 |
| step 2 | **CUDA OOM**, 95.00 of 95.22 GiB | 67.3 s, backward 46.539 |
| step 3 | — | 67.3 s, backward 44.776 |
| peak | **88.21 GiB** | **44.55 GiB** |
| median | none | **67.3 s/step** |

**1.98x on peak memory**, which is the 8-rows-versus-1 factor. micro=1 completes
3/3.

**The bucket hypothesis is refuted by the probe's own trace, not just displaced.**
Widths under micro=1 went **256 → 512 → 512** as rollouts grew 174 → 241 → 291
tokens. The width did widen, mid-run, and nothing OOMed. So a varying bucket width
is not sufficient to cause this; it may still contribute, but it is not the
binding operand.

## The comparison I withdrew comes back, valid this time

The snapshot withdrew a 1.94x against run 2's 229.2 s (wrong task, wrong bucket)
and then declined P1's 56.88 s/step as well, because 25.3 s was a warm *first*
step. Probe C fixes both halves: **67.3 s/step median over 3 steps against P1's
56.88 median — 1.18x**, same task (gsm8k), same group (8), same `micro=1`, and the
one remaining difference is cap **2048 vs 256**. That is a comparison worth having,
and it says the cap costs about 18% at this rollout length.

Cost of `micro=1`: **2.75x on the backward at the same width** (22.228 vs 8.070 at
width 256). So micro=0 is the faster setting that does not fit, and micro=1 is the
slower setting that does.

## What I destroyed while measuring

Run 2's own peak would have settled this in one line — if run 2 also peaked near 88
GiB it was always on the edge. It is unavailable: the entry records no peak figure,
and `runs/0f7006c74ea0/manifest.json` is gone from the pod because
`pod_sync.sh:28` runs `find . -mindepth 1 -delete` on the remote checkout before
untarring. My sync this tick deleted it. Line 12 rescues `bench-baseline.json`;
nothing rescues `runs/`, so every peer's run manifests die on the next sync by
anyone.

## Fix

`grpo-math-27b` already sets `micro=1`, so the recipe path is safe. What is not
safe is the CLI default: `--micro 0` at group 8 and cap 2048 does not fit one H20,
and nothing says so at the point of use. One line in `--micro`'s help, and
`pod_sync.sh` should exempt `runs/` from the wipe the way it already rescues the
baseline.

Neither is done here — this entry is the measurement, and both fixes are listed in
OPEN.md.

## Rule

**A shared shape needs every operand enumerated, not the ones both sides
remember.** "Exactly this shape — group 8, cap 2048, one H20" was true and still
compared two different configurations, because `micro` was in neither party's list.
Before attributing a difference to code drift, diff the *configs* field by field
against their definitions, not against a recollection of them.

Second: a mechanism can be sound and still not be the cause. The fragmentation
story explains the failure perfectly and the probe shows the width widening without
a failure. **Prefer the probe that varies one operand over the mechanism that
explains the observation.**

## Results

No runtime change. Measurement only.

| date | commit | machine | config | steps | peak | median |
|---|---|---|---|---|---|---|
| 2026-09-06 | faae3c8 | H20 card 6 | group 8, cap 2048, **micro=0** | **1, OOM at 2** | 88.21 GiB | — |
| 2026-09-06 | faae3c8 | H20 card 6 | group 8, cap 2048, **micro=1** | **3/3** | **44.55 GiB** | **67.3 s/step** |
| 2026-09-05 | 91977a8 | H20 card 0 | group 8, cap **256**, micro=1 | 100 | 39.78 GiB | 56.88 s/step |
| — | — | — | run 2's peak, for the one-line answer | 45 | **destroyed by `pod_sync.sh:28`** | — |
