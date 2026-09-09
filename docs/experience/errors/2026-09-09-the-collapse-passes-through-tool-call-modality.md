# The collapse passes through tool-call modality before going empty

Date: 2026-09-09
Status: open (next: advantage signs on mixed groups, steps 33-35)

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

Not established. What the run does and does not say:

- **λ=0, so an all-wrong group yields zero advantage, not negative** — the
  group-advantages guard zeroes a zero-variance group. In the all-wrong steps
  (34, 36+) the objective gives no signal at all, neither toward nor away from
  the tool-call modality.
- **In a mixed group, a 0-score row gets negative advantage, which pushes the
  policy *away* from that output.** If that were the dominant force here, the
  fragments would be suppressed; they instead reached 5/8 at step 35. The
  driver is elsewhere.
- **The reward matcher is modality-blind.** The step-28 row — a full
  `<tool_call>` wrapper around correct arithmetic — scored 1.0 because the
  boxed answer was present. The reward surface never distinguishes "wrong
  math" from "wrong modality": a malformed fragment scores 0 for lacking a
  boxed answer, same as any wrong plain-text rollout. Modality drift is
  therefore unpunished as long as it stays well-formed. That is a statement
  about the reward surface, not yet a mechanism for why the drift starts.

Missing measurement: the advantage signs on the mixed groups at steps 33-35.
Those rows are in the run's rollouts.jsonl; the signs were not checked.

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
