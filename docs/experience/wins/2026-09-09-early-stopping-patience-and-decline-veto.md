# Early stopping: patience in curve points, and a decline veto — 2026-09-09

**Status:** pending-remote. `src/tilerl/` runtime change on the train loop and the curve
path; the stop behavior only fires inside a real RL run, which this machine cannot run
(no GPU). Local evidence is the runnable check and its negative controls; the pod run is
the acceptance test.

## Context

#330 made the best curve point survivable (`adapter-best.safetensors`), but training
still ran to `--steps`: the snapshot saved the peak and the run still paid for the valley.
On seed 0's curve the run spent 17.4x the training time to end 1.4 pt lower than it had
already scored, and the per-question table showed 11 questions lost *after* the collapse,
independently of it — training past the peak is destructive, not just wasteful.

## What changed

`--patience N` (default 0 = never stop) early-stops the GRPO loop. Two stop paths, both
ruled by the paired 2xSE width #330 introduced:

1. **Patience.** A curve point that does not significantly improve on the best adds one
   to a count; a new best resets it; the run stops when the count reaches `patience`.
   The unit is **curve points, not steps** — one point is one `--eval-every` interval —
   and the arg help, the `EarlyStop` docstring and the manifest all say so, because the
   same patience means five different things at `--eval-every 25` and `5`.
2. **Decline veto.** A point significantly *below* the best (`score < best − 2·SE_paired`)
   stops the run immediately, spending no patience. The collapse is the event this
   feature exists for: seed 0's step 75 was −11.00 pt / 6.62σ, and under patience alone
   the run would have trained `patience` more points past it. Stopping never loses
   anything — the best snapshot is kept either way — and seed 0's recovery (412 → 456)
   still ended below the peak (467), so waiting for it bought less than keeping it.

`manifest["early_stopped"] = {at_step, kept_step, patience, reason}` with `reason`
`"patience"` or `"decline"`, so a later reader can tell a plateau walk-out from a
collapse. The default is off: flipping it on is a CHANGELOG default-flip event and waits
for the seed-1 verdict on whether the collapse reproduces.

3. **Startup refusal.** Early stopping is a verdict about the curve, so a run that cannot
   produce paired per-problem rows refuses at startup (`--patience > 0` with no
   `--eval-every` or no curve rows), and a point whose best's rows are missing or do not
   join refuses mid-run. The alternative was the silent fallback: the unpaired width is
   1.9x wider and never fires on a slow rise, and stopping with no width at all decides on
   noise. `patience=0` never asks, so it never refuses.

## Correctness evidence: three negative controls

`python -m tilerl.ledger` — four cells (plateau / plateau-then-rise / collapse /
missing-width refusal), each with a mutation verified red and reverted green:

| control | mutation | result |
|---|---|---|
| 1 | decline ignored (`if False and declined`) | red — the collapse cell no longer stops at the decline |
| 2 | stops on every new best | red — the plateau-then-rise cell stops before the rise |
| 3 | patience never counts (`stale` frozen) | red — the plateau cell never stops |
| 4 | refusal never raises (`if False: raise SystemExit`) | red — the missing-width cell runs to completion instead of refusing |

The cells pin all three cases: the stop-worthy stops, the not-stop-worthy does not, and
the immediately-stop-worthy is not slow.

## What the pod must show

A run with `--patience 1 --eval-every 5` on the GSM8K recipe stops at step 10 (the first
non-improving point after the step-5 peak), ships `adapter-best.safetensors` at step 5,
and the manifest's `early_stopped.reason` is `"patience"`. A collapse-shaped run stops at
the collapse point with `reason: "decline"` and `kept_step` at the pre-collapse peak.
