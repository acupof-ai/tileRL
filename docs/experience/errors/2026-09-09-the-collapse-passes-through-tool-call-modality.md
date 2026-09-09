# The collapse passes through tool-call modality before going empty

Date: 2026-09-09
Status: closed (observation; no fix)

## Context

The seed-0 control run (3276b687898d, patience=0, eval-every 5) collapses to
empty outputs over steps 33-36. Scanning its 800 rollout `text` fields for
tool-call syntax finds 6 rows, all at the collapse boundary:

- **step 28, g=4** — a complete `<tool_call>` / `<function=computer_use>`
  wrapper around correct arithmetic. Reward 1.0: the boxed matcher found the
  answer inside the wrapper. One rollout, still scoring.
- **step 33** — 7 of 8 rollouts empty; the one survivor is plain text, correct.
- **step 35** — 5 of 8 rollouts are tool-call fragments, all reward 0:
  - `<tool_call>` alone (11 chars),
  - `<tool_call>\n<function>\n<parameter>\n<parameter=math>...` (malformed),
  - `<function=computer_use>` with `action: "think"`, `action: bash`,
    `function=problem sol` — three different invented tool schemas in one step.
- **step 36 onward** — all 8 rollouts empty, every step, to step 100.

The modality drift is a one-step transition state between "mixed empty"
(step 33) and "fully empty" (step 36), not a stable alternative mode.

## Root cause

Hypothesis, not measured: the policy under RL pressure drifts into a mode the
base model knows (Qwen3.8's chat template has tool-call modes; `<tool_call>`
is single token 248058). The reward matcher only reads boxed answers, so
malformed tool-call text scores 0; those rollouts draw negative advantage,
which reinforces the drift toward the zero-reward absorbing state (empty
output). The step-28 row shows the matcher is not fully blind to the modality
— a well-formed wrapper around a boxed answer still scores — so the
self-reinforcement claim needs the advantage signs, which were not checked.

## Fix

None. Diagnostic value: tool-call fragments in rollouts are an early-warning
signature of an in-progress empty-output collapse, appearing one step before
the curve moves (step 35 fragments, curve 68 at step 35 vs 86 at step 30).

## Rule

Two limits on what this observation can compare against:

1. The surviving run (86a06dc8c420, bd72288) predates #328, so its rollouts
   store no decoded text — whether it emitted tool calls is unknowable from
   stored artifacts. "0 hits" there is an unavailable measurement, not a zero.
   A modality audit only covers runs after #328 landed.
2. The collapse entry this feeds is held until the step-3 experiments
   (with/without before-arm eval) land — the fragments characterize the
   collapse, they do not attribute it.

## See also

[The dip hit problems a healthy seed solves](2026-09-09-the-dip-hit-problems-a-healthy-seed-solves.md)
— the same run pair, dip mechanism and the same-batch floor correction.
