# tied_correctness: the validity gate's real input — 2026-09-09

## Context

The P1 validity gate reads `tied_group_fraction` from the manifest. `tied`
(train.py) is the fraction of groups whose 8 rewards are *exactly* equal. At
λ=0 (binary rewards) this is informative: an all-correct group ties and
carries no gradient. At λ=0.1 the length-aware reward is continuous
(`correctness − λ·tokens/cap`), so eight rollouts never produce identical
rewards — `tied` is 0.00 at every step by construction. The gate cannot turn
red. Run `d6447c0abe5b` (MATH L5, λ=0.1) reported `tied=0.00` for 10 straight
steps; the gate was green throughout, and the run's groups were 60% tied at
the correctness level (6/10 all-correct or all-wrong).

## What changed

`grpo_loop` takes an optional `correctness_fn(prompt, completion) -> float`
that returns binary correctness before the length term, tiebreak, and live
mask. When given, the loop computes `tied_correctness` — the fraction of
groups whose binary correctness is all-same — and yields it as the 8th tuple
element. `cli.py` logs it (`tied_c`) and records it in the manifest as
`metrics.tied_correctness`.

The quantity is comparable to a λ=0 run's `tied` fraction: both measure "was
this step gradient-free at the correctness level." The roadmap's 0.650
(λ=0, cap 2048, 32% truncation) is the reference point; this run measured
0.60 at λ=0.1, cap 6144.

## Rule

A gate's field must be able to turn red under the condition it guards.
Verify the field can change before trusting the gate. `tied` at λ>0 is a
field that cannot change; `tied_correctness` is the field that can.
