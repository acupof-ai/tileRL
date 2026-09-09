# Group-chat RL: a survey and a choice — 2026-09-09

Written because ckl forked `OpenBMB/Meshy` to `cklxx/Meshy` (2026-09-09) and asked how
group-chat / multi-agent collaboration would do RL. Same spec as the architecture survey
([design-rl-architecture.md](design-rl-architecture.md)): survey, then a choice with
explicit adopt/reject criteria. Research only — no `src/` changes.

## The two repos

`cklxx/Meshy` **is** `OpenBMB/Meshy`: GitHub marks it a fork, created 2026-09-09, with
**zero divergent commits** (`git log upstream/main..HEAD` is empty). It is the same
codebase ckl is reading, not a different project with the same name — the same-name trap
that produced eight errors on 2026-09-08 does not fire here, and the check that settles it
is the fork graph, not the name.

A prior look at this codebase (2026-09-08, internal research) rejected it as an async-RL
infrastructure play: the upper bound on wall-clock gain was 1.36x
(`max(63.156, 22.459 + the overlap)`, against an 85.617 s step), and
[design-rl-architecture.md](design-rl-architecture.md) already chose one process and one
weight copy over Meshy's service topology. **That rejection was about async rollout, not
about group chat** — the question here is new and the old conclusion does not transfer.

## The core distinction: two meanings of "group"

GRPO's group is **N independent rollouts of one prompt**. The rollouts are causally
unrelated samples from the current policy, so each is a control for the others, and the
group mean is a Monte Carlo estimate of the policy's expected reward on that prompt. That
is the entire statistical content of `(r − mean)/std`.

A group chat's group is **N agents in one conversation**. Agent i's output is conditioned
on agents 1..i−1's outputs. The members are not controls; they are a causal chain.
Subtracting a baseline across the members of one chat — porting `group_advantages`
([train.py:277](../src/tilerl/train.py)) over the N agents in an episode — is subtracting
a baseline from a set of causally dependent quantities, and the result has no
interpretation as a policy-gradient estimate.

## Q1 — How credit assignment lands on the seam

The seam survives, but **the group's meaning has to change**: the group becomes K
independent EPISODES of the same chat scenario (K re-runs with different sampling seeds),
and the advantage is computed per agent-position across episodes:

```
advantage(episode k, position i) = (r(k, i) − mean over K episodes at i) / std
```

Three cases, in order of how well they fit:

1. **Per-agent rewards exist.** A judge scores each agent's contribution — the tree
   already has this machinery ([judge.py](../src/tilerl/judge.py), pairwise verdicts to
   GRPO advantages, stage 4(b)). Each position gets its own reward, its own baseline over
   K episodes. This is GRPO with the prompt replaced by a scenario and the rollout
   replaced by an episode. Clean fit.

2. **Joint outcome only** (one reward for the whole chat). Every position in an episode
   gets the same advantage, so within an episode there is no differentiation at all — the
   gradient says "be more like whatever happened in winning episodes", uniformly. The known
   fix is difference rewards / COMA: credit each agent against a counterfactual episode
   outcome, which needs a model of the counterfactual and is a research program, not a
   seam change. Without it, joint-reward group chat is repeated-play GRPO with a longer
   prompt.

3. **Distinct agent weights** (different models or different LoRAs per role). This is a
   new product: N policies, N advantage streams, N optimizers. Nothing in the tree
   supports it, and it is not what a one-card runtime is for.

**The minimal honest observation:** single-model self-play with a joint reward is GRPO
with a chat-shaped, multi-turn prompt. The engine already interleaves decode streams
(continuous batching); the rollout loop would feed agent i's message back as agent i+1's
prompt; the tape would mask the loss to the trainable model's tokens. The group is still K
repeated episodes. Multi-agent becomes a NEW credit-assignment problem only when agents
have distinct weights or distinct reward streams — cases 1 and 3, not the self-play that
"group chat" first suggests.

## Q2 — What Meshy does and does not do

Read in the code, not the README:

- **No multi-agent anything.** The only "multi-turn" in the tree is tokenization: chat
  templates put an EOS after every turn of a single rollout's prompt history
  (`backend/titan/batch.py:194`). `group_size` rollouts per prompt
  (`service/rollout.py:57`), same as any GRPO engine.
- **Its advantage functions are the same seam as ours.** `advantage.py` does
  `reward − mean` over a prompt's rollout group — it even drops the std in its DAPO path
  — plus DAPO length shaping. There is no mechanism here for causally chained actors.
- **What it is:** a service-per-layer async RL engine (inference / training / rollout as
  processes over a TransferQueue). That is the infrastructure the 2026-09-08 review
  already rejected at 1.36x, and it is orthogonal to group chat.

**Meshy contributes nothing to this question.** It neither solves group-chat credit
assignment nor blocks it; it is the same single-agent group-rollout design we already
run, with a process topology we already chose against.

## Q3 — Tied groups: more common or less?

The tied fraction is set by the **cross-episode variance of the episode outcome**. The
chat changes that variance in two opposite directions:

- **Less variance (more ties):** agent 2 sees and can correct agent 1's error, so the
  joint outcome is buffered against individual mistakes — a Condorcet-style effect when
  agents are better than random and can check each other.
- **More variance (fewer ties):** a chat adds coordination failure modes that do not exist
  in one rollout — misunderstanding, derailing, one agent dominating — and a longer
  trajectory has more tokens at which to diverge.

Which force wins is a measurement, and this doc will not invent a number. But one claim
needs no measurement: **for a Bernoulli episode outcome the all-success tied fraction is
monotone in the success rate** — it rises as p → 1. The entire motivation for multi-agent
is a higher success rate on the same task, so *if group chat delivers its motivation, it
makes tied groups more common*, not less. GSM8K already sits at `tied_group_fraction`
0.81 (73/81 steps tied at the ceiling); a stronger joint system sits higher.

There is also a structural trap in the counting: the N agents in one episode are
correlated by construction — they share one conversation — so the baseline's effective
sample count is K (episodes), never N×K (agent-positions). Counting each agent-position as
a sample is the population error in a new costume: the values are not drawn independently,
they are one trajectory's worth.

The escape is the same one the roadmap already names for single-agent GRPO: harder tasks
(MATH-level), where the success rate leaves the ceiling. Group chat does not dodge the
tied-group problem; at its best it makes it worse, and it inherits the same fix.

## Relationship to `time_to_score`

```
time_to_score = steps_to_score  x  seconds_per_step
```

Group-chat RL shortens neither factor for the current objective:

- **`seconds_per_step` rises.** An episode is N agents × multiple turns — strictly more
  generated tokens than one GSM8K rollout. The step gets longer.
- **`steps_to_score` is not demonstrably shortened.** The score is still single-agent
  GSM8K accuracy; chat RL is an indirect training method whose reward signal is diluted
  across agents and whose tied groups are more common (above), so each step carries fewer
  effective gradients. No measurement exists, and the direction is not favorable.

**This is a different product line, not a lever on the objective.** It trains agents for
chat-task performance, which is a new score. Whether to open that line is ckl's call; the
doc's job is to say plainly that it should not be sold as a `time_to_score` improvement,
the way the 1.36x async play was evaluated and rejected on exactly this ground.

## The choice

**Reject group-chat RL for the roadmap.** The criteria:

- It does not touch `time_to_score` — both factors move the wrong way or are unmeasured.
- The one external artifact in question (Meshy) has no multi-agent machinery; its
  infrastructure was already rejected at a measured 1.36x bound.
- The tied-group failure mode, already the binding constraint at 0.81, gets structurally
  worse if multi-agent delivers its only motivation.

**Adopt criteria for the future** — all three required, in order:

1. A chat task is in the objective, set by ckl — not inferred from a fork.
2. A per-agent reward stream (the judge path, case 1) or a repeated-episode design with
   K ≥ 4 and an outcome that is not tied. Joint-reward self-play without this is
   longer-prompt GRPO and should be evaluated as that, not as a new method.
3. A measurement that the chat system's success rate beats the single agent's on the same
   task. Without it, multi-agent is cost with no benefit — the same bar the architecture
   survey set for every infrastructure play.

When adopted, the minimal path is small and needs no engine change: self-play on one
engine, K repeated episodes as the group, per-position advantage across episodes,
`judge.py` for within-episode differentiation, per-turn loss masking on the tape. The
seam was designed for this; the group's meaning is the only thing that has to move.

## Sources

- `cklxx/Meshy` @ 8c8ec69 (= `OpenBMB/Meshy` main, zero divergent commits):
  `meshy/advantage.py`, `meshy/service/rollout.py`, `meshy/worker/rollout.py`,
  `meshy/backend/titan/batch.py`
- [design-rl-architecture.md](design-rl-architecture.md) — the one-process choice and the
  1.36x async bound
- [train.py:277](../src/tilerl/train.py) `group_advantages`, [judge.py](../src/tilerl/judge.py)
- 2026-09-08 internal review of `OpenBMB/Meshy` (async RL, rejected; no tree entry)
