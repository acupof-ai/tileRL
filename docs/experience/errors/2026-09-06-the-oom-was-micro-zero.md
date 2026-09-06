# The OOM was `micro=0`, not the buckets — 2026-09-06

> Status: Shipped. `--micro` defaults to 1 and `pod_sync.sh` exempts `runs/`. Both
> OPEN.md rows removed.

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

## The wipe destroyed nothing, and that is not the point

I wrote that my sync deleted `runs/0f7006c74ea0/manifest.json`. **It did not exist.**
Per tilerl-25, who ran it: run 2 died on SIGTERM at step 45, and `write_manifest`
lived only inside `_finish` until `8388cbf` ("write the manifest before the loop,
not only in _finish"), so run 2 never wrote a manifest or a `rollouts.jsonl` at all.
I could not confirm the ordering from the tree — run 2's entry records a run id and
no commit sha, so this rests on 25's account of their own run plus the existence of
that commit, not on a check I ran.

So run 2's peak is unavailable because it was never written, not because I deleted
it. My claim to have destroyed it was wrong in the direction that made my own error
look worse, which is the one direction that does not get re-checked.

**The real point is forward-looking.** Since `8388cbf` every run writes its manifest
*before* the eval arms, so a killed run now leaves one — and `pod_sync.sh:28`'s
`find . -mindepth 1 -delete` would delete exactly the evidence that commit exists to
preserve. The next killed run is the one that loses its manifest, and any peer's sync
does it.

Measured per implementation, because **the two `find`s do not agree and I first
published the table as if they did**:

| wipe spelling | GNU 4.9.0 (pod, ubuntu CI) | BSD (macOS) |
|---|---|---|
| `find . -mindepth 1 -delete` (current, control) | `runs/` **GONE** | `runs/` **GONE** |
| `-path ./runs -prune -o -delete` | **refuses**, rc=1, deletes nothing | `runs/` **GONE**, rc=0, silent |
| `! -path './runs' ! -path './runs/*' -delete` | **KEPT**, rest gone | **KEPT**, rest gone |

Row 3 — the shipped spelling — is the same on both, which is what the fix needed.
Row 2 is where they diverge: **`-delete` implies `-depth`, which disables `-prune`**,
and GNU says so in its own words (`the -delete action automatically turns on -depth,
but -prune does nothing when -depth is in effect`) and exits 1 without deleting, while
BSD accepts the expression and deletes `runs/` with no complaint.

**My first table reported the BSD consequence as the mechanism's consequence**, on a
line that also claimed the pod's `find` had confirmed it. Two errors in one row: I ran
the Mac probe through a shell function that resolves to `bfs`, not `find` at all, and
the arm I did run on the pod was the shipped spelling, never the `-prune` one. The
ubuntu CI leg of my own PR is what caught it — `test_prune_does_not_protect_runs`
asserted `runs/` was deleted, which is false on GNU. The test now asserts what holds on
both: `-prune` never yields a working exemption. The mechanism claim stands; the claim
that it silently deletes does not, off macOS.

Not a risk, checked: `pod_sync.sh:12`'s rescue is not a file copy that could flatten
a tree. `baseline.py pull` `cat`s one remote JSON and merges its keys into the local
file, so exempting a directory has nothing in common with how the baseline is
rescued.

## Fix

Both landed, and the first is a **default flip**, not a warning.

**`--micro` defaults to 1.** Both shipped RL recipes already set 1
(`recipes.py:21`, `:37`); the value 0 does not fit the production card at the
production shape, and a default that OOMs on step 2 is the wrong default. 0 stays
available for anyone who has measured that it fits. Nothing in the tree depends on
the old default: `test_rl.py:253` passes 0, 1 and 3 explicitly and asserts they land
on the same weights to within 1e-6, so this flips which arm is default without
changing what any arm does.

**`pod_sync.sh` exempts `runs/`** with `! -path './runs' ! -path './runs/*'`, the
spelling the table above shows actually works.

The OPEN.md rows for both come out in the same change.

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

Runtime change: the `--micro` default. Probe C's two arms are its measurement — the
new default is row 2, the old one is row 1.

| date | commit | machine | config | steps | peak | median |
|---|---|---|---|---|---|---|
| 2026-09-06 | faae3c8 | H20 card 6 | group 8, cap 2048, **micro=0** (old default) | **1, OOM at 2** | 88.21 GiB | — |
| 2026-09-06 | faae3c8 | H20 card 6 | group 8, cap 2048, **micro=1** (new default) | **3/3** | **44.55 GiB** | **67.3 s/step** |
| 2026-09-05 | 91977a8 | H20 card 0 | group 8, cap **256**, micro=1 | 100 | 39.78 GiB | 56.88 s/step |
| 2026-09-05 | — | H20 | MATH, cap 2048, micro=1, **1434-token rollouts** | 45 | never written | 229.2 s/step |

Row 4 is a different task at 4.9x row 2's rollout length. **67.3 against its 229.2 is
not a speedup** and nothing may cite it as one; the only valid pair in this table is
rows 2 and 3 (1.18x, cap 2048 vs 256). Its peak is blank because run 2 never wrote a
manifest, per the section above.
