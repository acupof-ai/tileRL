# What a length term in the GRPO reward can and cannot recover — cpu, 2026-09-06

> Status: analysis only, no code. Bounds the third fix in
> [OPEN.md](../OPEN.md) before anyone writes it.

## Context

[Run 2](../errors/2026-09-06-the-rollouts-grew-into-the-cap.md) lists three
fixes; two have landed (#140 padding buckets, #143 periodic guard) and the third
— "a length term in the reward, or a length-aware advantage" — is called "the
actual cause". It is unclaimed and carries no number. Everything below comes
from the entry's own 45-step table and from reading `group_advantages`; no GPU
was used.

## What Worked

**42.2% of run 2's steps produced no gradient at all**, and that is the quantity
a length term is supposed to attack. 19 of 45 steps had `tied == 1.00`, costing
**1.08 h of 2.71 h (39.8%)** of wall clock. The split matters: **5 floor ties**
(reward 0.000, every rollout wrong) and **14 all-pass ties** (reward 1.000,
every rollout right). `tied` alone cannot be a trigger for anything, because
those 14 healthy steps are indistinguishable from the 5 damaged ones in it.

**A length term reaches at most 17 of the 19, and provably cannot reach 2.** A
tie means all 8 rewards are equal; a length-aware reward breaks it only if the 8
completions differ in length. At steps **41 and 44** the group mean equals the
cap exactly (2048), so all 8 rollouts are clipped to the same width and *any*
function of length is constant within the group. Those two steps are recoverable
only by raising the cap, never by reweighting.

**λ is not a taste parameter; it has a closed form.** A wrong answer outranks a
right one iff `λ·(len_right − len_wrong)/cap > 1`, so ordering is preserved for
all `λ < cap/gap`. The worst reachable gap is `cap − 1`, which gives

| worst length gap | ordering safe for |
|---:|---|
| 2047 (= cap − 1) | λ < 1.000489 |
| 1500 | λ < 1.37 |
| 1000 | λ < 2.05 |

so **λ ≤ 1.0 never lets a shorter wrong answer outrank a longer right one, for
any lengths whatsoever**. Bisecting the worst case (one correct rollout at the
cap, seven wrong at length 1) puts the first inversion at **λ = 1.000488550**
against a closed form `2048/2047 = 1.000488520` — agreement to eight digits, so
the bound is exact rather than approximate.

**The boundary has to be probed at the gap that defines it, not by sampling.** A
random search over uniform lengths reports 0 inversions at λ = 1.001, which
looks like the bound is loose; it is not. An inversion needs a gap near the cap,
and 8 uniform draws produce a gap above `0.99·cap` in **0.260%** of groups
(519/200 000) — and each of those must additionally have the *long* rollout
correct and a short one wrong. A 50 000-group search expects ~130 candidates
before that second condition, so its silence measures the sampler, not λ. The
first version of this entry cited "114 inversions at λ = 1.01" from exactly such
a search and called it the boundary; that number was luck.

**On an all-pass group λ cancels completely.** `group_advantages` divides by the
group's std, so when correctness is constant the advantage depends only on the
*ordering* of lengths — λ = 0.1 and λ = 1.0 produce the identical vector
`[1.29, 0.80, 0.62, 0.38, 0.18, −0.27, −0.94, −2.05]`. λ therefore does nothing
on 14 of the 19 tied steps, and tunes only the correctness/length trade-off on
mixed groups. Anyone tuning λ against the tie fraction will see no movement and
conclude the term does not work.

## The claim this does not support

The entry reads its own table as "reward is 0.893 for steps under 1200 tokens
and 0.458 at or above, so short rollouts score nearly twice as well and the
policy still moves toward long ones". Both numbers reproduce exactly (n = 21 and
24). But that is a **cross-step** correlation and GRPO's advantage is **within a
group, on one prompt**. Prompt difficulty confounds it: a harder problem yields
both longer completions and lower reward. Each of the 45 steps drew a different
prompt, so nothing in this table separates length from difficulty, and the
mechanism sentence is an inference rather than a measurement. The fix may still
be right; the evidence offered for it is about the wrong axis.

## Rule

Bound a proposed reward term by what it can reach before implementing it: count
the steps it cannot touch, and derive the coefficient's safe range from the
ordering it must preserve rather than sweeping for it. When the advantage is
std-normalised, check whether the coefficient survives normalisation at all —
here it vanishes on 14 of 19 target steps.

**And probe a derived boundary at the input that defines it.** A random sweep
around λ = 1.0 reports no inversions because the configuration that inverts
occurs in 0.26% of samples; the sweep's silence reads as confirmation and is
measurement of the sampler. Bisect on the worst case instead.

## Results

No runtime change, so nothing to bench. Reproduce with the table in
`errors/2026-09-06-the-rollouts-grew-into-the-cap.md` and `group_advantages`
(`src/tilerl/train.py:258`).
