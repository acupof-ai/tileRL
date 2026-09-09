# The step-75 dip hit problems a second seed solved at the same step

Date: 2026-09-09
Status: closed (instrument deferred, not refuted)

## Context

Run 86a06dc8c420 (seed 0) dipped from 467/500 at step 50 to 412/500 at step 75,
then recovered to 456 at step 100. Two mechanisms were on the table:

1. **Problem nature** — the dip hit intrinsically fragile problems that any
   policy loses around this training stage. Early stopping cannot save them.
2. **One bad update** — the dip destroyed problems the policy could solve.
   Early stopping is the cure.

A second seed (run d183f74d233c, seed 1, same recipe) distinguishes the two.
All numbers below come from per-problem eval-curve rows paired by eval-file
row (`scripts/curve_table.py`, in this PR).

## What the second seed says

Three independent statements; do not merge them.

**1. Seed 1 did not dip.** 472/471/474/474 at steps 25/50/75/100
(94.4/94.2/94.8/94.8%). Every adjacent pair is within noise. The dip is not a
property of the recipe at step 75.

**2. The dip destroyed problems a healthy policy solves.** Of seed 0's 88
wrong answers at step 75, **72 are problems seed 1 answers correctly at the
same step (82%)**. Only 16 of the 88 overlap seed 1's wrong set. The dip's
main body is not "problems hard for both seeds". This statement needs no
baseline and no calibration; it stands on its own.

**3. The z instrument was NOT APPLIED.** The three-band verdict
(problem nature / bad update / inconclusive) was calibrated on
**healthy–healthy** overlaps: two healthy seeds' wrong sets intersecting,
z@25 = +12.4 and z@50 = +10.8 (n=500 both). At step 75 seed 0 had already
dipped, so the only available pair is **post-dip–healthy** — a different
quantity from the calibration. Running it anyway gives z@75 = +6.0
(obs=16, E=4.58, max=26, n=500), but reading that against the healthy bands
would be comparing a number to a population it was never drawn from. The
instrument stays in `scripts/curve_table.py`, ready for a dip that
reproduces in a second seed.

## Instrument validation

So the deferred application is trustworthy when it comes:

- The gold-answer tripwire is blind to same-gold pairing errors — GSM8K has
  195 distinct golds in 500 rows, so a same-gold wrong row passes silently.
- `--negctl same-gold` remaps the pairing within same-gold rows (fixed seed).
  It passes the tripwire **by construction** and drops the step-25 z from
  +12.4 (n=500) to +2.4 (n=374; the mutant is not a bijection, so the paired
  population shrinks — the two z's carry their own n).
- The z sees the pairing error; the tripwire does not. The residual +2.4 is a
  speculative reading (difficulty shared within same-gold problem families),
  not a measurement — a true-random-pairing control would be needed to measure
  it, and it changes no verdict.

**Resolution floor.** A same-policy retest flips 10.4% of per-problem
outcomes: 52/500 between two greedy evals of the same step-100 weights,
differing only in batch composition, with a net score difference of 2
(measured in [the companion entry](2026-09-09-the-collapse-did-not-replicate-the-plateau-did.md)).
The healthy–healthy z@25 = +12.4 carries this noise on both sides, so it
remains a valid baseline, but the instrument cannot resolve per-problem
structure below the ~10% flip rate. In the plateau this binds: the cross-step
policy difference (50 flips over 25 steps) is the same size as the
same-policy retest noise (52).

## What this does NOT establish

- **The dip's occurrence rate is unmeasured and unmeasurable** at current
  machine budget. One dip in two runs gives an interval near [0.01, 0.99];
  "rare" or "isolated" is not supported. The pre-registered "seed 1 monotone
  or plateau ⇒ one bad update" was too loose: non-reproduction excludes only
  determinism, not a 20% rate.
- **Early stopping does not rest on dip reproduction.** Seed 0 met the exit
  score at step 5 (471/500 = 94.2%) and gained nothing net by step 50
  (471→467, peak 473 at step 15). Seed 1 gained nothing net from step 25 to
  100 (472→474). Both curves agree the extra steps bought ~0; they disagree
  on the dip. The plateau is the early-stop case, and it is consistent across
  both seeds.

## Rule

A verdict calibrated on one population does not transfer to another — label
the population with the number. The z bands were calibrated on healthy–
healthy overlaps; a post-dip–healthy z is a different instrument reading, and
forcing it into the bands reads a threshold out of a range the data cannot
resolve. Register the "instrument does not apply" outcome before the data
lands, or the number will be read into whichever verdict fits.

## See also

[The collapse did not replicate; the plateau did](2026-09-09-the-collapse-did-not-replicate-the-plateau-did.md)
— the plateau and its pricing (the early-stop case), from the same two runs.
This entry is the dip's mechanism; that one is what the plateau buys.
