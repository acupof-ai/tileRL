# The first steps-to-target numbers, and the validity gate that voided the headline one — 2026-09-08

> Status: **B landed, A voided by its own gate.** The project measures `seconds_per_step`
> (85.617 s, #273) and had never measured `steps_to_score`, so every infrastructure
> optimization multiplies an unknown. This is the instrument for that factor. It produced a
> usable negative result and refused to produce the positive one, for a reason it measured
> itself. `scripts/steps_to_reward.py`.

## What was asked

`design-rl-stack.md:27` reports ISO reaching AdamW's endpoint in **~2.7x fewer steps** on
Qwen3-4B/8B under RLVR, and `:34` states the consequence correctly: 2.7x fewer steps is 2.7x
less rollout, the denominator lever that also saves the numerator. **Nobody has verified it
once in this setup.** The ask: the first steps-to-loss curve at any scale, minutes not card
hours, on the argument that a 2.7x that fails to reproduce small is worth knowing before
anyone spends pod time.

## B: on an uninformative spectrum ISO is a net cost, 0.75–0.81x

Steps for each optimizer to first reach a target loss, tiny model, CE objective, each arm's
learning rate swept separately and its best taken (a fixed lr measures the tuning, not the
optimizer):

| arm | loss ≤ 10 | loss ≤ 8 | loss ≤ 7 |
|---|---:|---:|---:|
| Adafactor lr=0.1 | **7.3 ± 0.5** | **9.0 ± 0.8** | **12.7 ± 1.2** |
| ISO(Adafactor) lr=0.1 | 9.3 ± 0.5 | 12.0 ± 0.8 | 15.7 ± 0.9 |
| ratio | 0.79x | 0.75x | 0.81x |

Four learning rates per family (3e-3, 1e-2, 3e-2, 1e-1), three seeds each; every lr favours
Adafactor, so the sign is not a tuning artefact. **ISO needs 1.25–1.33x MORE steps here.**

**This does not refute the 2.7x, and the reason is the mechanism.** ISO's premise is that
RLVR preserves the base model's singular spectrum, so it freezes Σ₀ *from the base*. On a
`build_random` init there is no informative spectrum — freezing it locks a random constraint,
and ISO is a pure constraint cost. The useful corollary is narrower and firm:

**No ISO smoke test on a random initialization can predict the 2.7x, whichever observable it
uses.** That lands directly on `tests/test_iso.py`'s "SFT loss falls", which is P3's own exit
criterion, and it is the second defect found in that assertion today (the first being that it
passes with ISO's 2D path disabled, `tilerl-48`'s #303).

## A: an SFT'd base, and the gate voided it

Freezing a random spectrum is the flaw, so the fix is to SFT first and put information in Σ.
Whether tiny's SFT spectrum resembles a pretrained one is **not** an unknowable ceiling — it
is measurable inside the experiment, which is `tilerl-27`'s framing and the reason this run is
worth reporting at all. ISO's premise is a preserved spectrum, so record Σ's drift during the
RL phase.

40 SFT steps (loss 23.734 → 5.966, fresh batch each step), then 24 GRPO steps from that base:

| arm | reward | Σ drift max | Σ drift mean |
|---|---|---:|---:|
| Adafactor | 0.639 → 1.000 | **22.06%** | 3.51% |
| ISO(Adafactor) | 0.639 → 1.000 | 2.61% | 0.22% |

**The premise fails here.** A free optimizer moves the spectrum 22% at its worst matrix during
RL, so this fixture is not a setting where "RLVR only rotates the frames" holds, and a step
ratio measured in it does not carry the paper's claim.

**Which arm's drift tests the premise is the part worth stating.** ISO's own 2.61% is not
evidence about the spectrum — ISO freezes Σ by construction, so its drift is the *floor of the
measurement's precision* (polar retraction and dtype round-trips), not a finding. Reading it
as "the spectrum barely moved, premise confirmed" would be measuring the instrument. Only the
**free** optimizer's drift is informative, and it says no.

**A second, independent reason the ratio is void:** the reward saturates. Over 12 steps at
group 6 with 6-token completions:

```
0.167, 0.194, 0.500, 0.833, 0.833, 0.833, 0.833, 0.833, 0.833, 0.833, 0.833, 0.833
```

Flat from step 4. A half-vocabulary token-rate reward on 6 tokens has a ceiling the policy
reaches almost immediately, so "steps to target reward" is either 1 step or never for every
target — visible in the table the script prints, where two of three targets resolve at 1 step
for both arms. **Two independent voids, either of which is sufficient**, and the second would
have been invisible had the first not made me look at the reward trace.

## What survives

The instrument. `steps_to_reward.py` produces steps-to-target with error bars from a swept
learning rate and refuses its own output when the premise it depends on fails. The shape
transfers unchanged to a real model and a real reward; what has to change is the fixture, and
the two things it needs are now named rather than guessed:

1. **A base whose spectrum carries information** — which means a real pretrained checkpoint,
   because 40 SFT steps on tiny does not produce one (measured: a free optimizer still moves Σ
   22%).
2. **A reward with headroom** — one the policy cannot saturate in 4 steps. GSM8K correctness
   qualifies; a token-rate on 6 tokens does not.

Both need the card. The CPU half of this question is answered: **it cannot be answered on
CPU**, and that is a cheaper thing to know than a curve nobody could interpret.

## Rule

**A validity gate has to name which arm tests the premise.** The obvious reading — "ISO's Σ
barely moved, so the spectrum is preserved" — is circular, because ISO freezing Σ is the
mechanism, not the observation. The free arm is the only one whose drift is data. A gate
pointed at the wrong arm would have passed and licensed the ratio.

**Suspect the instrument when a number contradicts a published one by 3.5x, and let the
suspicion name the setting rather than the precision.** 0.79x against a reported 2.7x is not a
precision problem; it is a different mechanism, and the difference is stated in the paper's own
first sentence. The rule that catches this is reading the claim's *conditions* before
measuring against it.

**A defect found in your own instrument is the cheapest possible moment to grep the tree for
its shape.** My first steps-to-loss script fed one fixed batch repeatedly, so targets were
reached in 3–5 steps and any ratio was noise. I fixed it, kept going, and needed a peer to
point out that `tests/test_iso.py:80` has the identical line — gating a roadmap phase exit. A
bug you just debugged is the one you can recognize fastest.

**Two voids are better than one.** Once the Σ gate fired I looked at the reward trace and
found saturation, which no amount of spectrum work would have fixed. A single sufficient
reason to distrust a measurement invites patching that one reason; looking for the second is
what tells you the fixture is wrong rather than one knob.
