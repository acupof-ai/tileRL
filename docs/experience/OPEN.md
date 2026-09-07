# Open defects

Documented in `errors/` with a fix named and not landed. A PR that lands the fix removes
its line. Writing the entry and adding the line are one act — an entry with `Status: open`
and no line here is the same defect this file exists to stop. Reviewers check a change on the same path against this list.

| entry | path | fix |
|---|---|---|
| [a miss self-reinforces](errors/2026-09-07-a-miss-self-reinforces.md) | `src/tilerl/engine.py:1031` `_finish_prefills` | one 31k-token miss published 62 chunk entries into a 6-snapshot budget and evicted every other session's shared head, so 11 of 12 sessions missed in sequence at 14.1 s each. **Fixed at the publisher** — the first interior boundary plus the last, a constant 2 publishes at any prompt length ([wins/2026-09-08](wins/2026-09-08-cut-the-prefill-publish-flood.md)). The remaining open half is the **V100 alternation** the cell was run to check: turn-0 hits on every other conversation, which did not reproduce on the H20 and needs the V100 grid with the per-row instrument |
| [the rollouts grew into the cap](errors/2026-09-06-the-rollouts-grew-into-the-cap.md) | `src/tilerl/eval.py` `MATCHERS`, GRPO advantage | a length term in the reward or a length-aware advantage — `boxed_match` is correctness-only, so nothing prefers the shorter of two correct answers and the policy lengthens until it truncates |
