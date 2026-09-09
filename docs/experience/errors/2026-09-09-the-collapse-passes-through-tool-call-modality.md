# The collapse passes through tool-call modality before going empty

Date: 2026-09-09
Status: open (next: the trigger — first empty completions at step 32; the mask below explains absorption, not onset)

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

The absorption into empty outputs is the advantage mask, measured and
code-verified. The trigger is still open.

`group_advantages` masks zero-length completions out of the group statistic:
they set neither the mean nor the std and get advantage 0 (the `live` mask,
`train.py:282`; call site passes `live=[len(c) > 0 for c in comps]`). The
measured signs, steps 28-36:

- **step 32** — 2 empty, 4 wrong-text, 2 correct. Empties +0.000, wrong-text
  -0.707, correct +1.414. **Empty beats wrong-text at the margin**: the
  objective ranks producing nothing above producing a wrong answer.
- **step 33** — 7 empty, 1 correct. The correct row is the only live one, so
  the live std is 0 and its advantage is +0.000 too. A mixed-reward group is
  silent because the mask isolated the one row that could carry signal.
- **step 35** — 5 tool-call fragments (live, all reward 0) + 3 empty. All
  +0.000: zero variance among the live rows.

So every step that produces some empties makes "produce nothing" the
best-advantaged action in hindsight, and the group goes silent once empties
dominate. The mask is the absorbing state.

What this does not explain is the **onset**. The first empties appear at step
32, and the mask is identical in the surviving run. Why this trajectory
started producing empties is the open question — the step-3 divergence (same
seed, different sha, diverging before any eval) and the check-3 / A-B
experiments address it.

The positive-reinforcement path — a tool-call row scoring 1.0 through the
modality-blind matcher and then drawing positive advantage — never fired
after step 28. Step 28's tool-call row sat in an all-correct group (advantage
0); every later fragment scored 0.

## Fix

None. Diagnostic value: tool-call fragments in rollouts are an early-warning
signature of an in-progress empty-output collapse, appearing one step before
the curve moves (step 35 fragments, curve 68 at step 35 vs 86 at step 30).
They are cheap to detect in rollout token ids: `<tool_call>` is a single
token, id 248058 (checked on the pod with
`AutoTokenizer.from_pretrained(TILERL_QWEN38_SOURCE).encode("<tool_call>")`
→ `[248058]`).

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
