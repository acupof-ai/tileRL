# Open defects

Documented in `errors/` with a fix named and not landed. A PR that lands the fix removes
its line. Writing the entry and adding the line are one act — an entry with `Status: open`
and no line here is the same defect this file exists to stop. Reviewers check a change on the same path against this list.

| entry | path | fix |
|---|---|---|
| 2026-09-06-the-rollouts-grew-into-the-cap.md | src/tilerl/train.py (batch width) | pad the rectangle to a power-of-two bucket, not the cap |
| 2026-09-06-the-rollouts-grew-into-the-cap.md | src/tilerl/cli.py (_refuse_short_rollouts) | re-check rollout mean against the cap every N steps, not once |
| 2026-09-06-the-rollouts-grew-into-the-cap.md | src/tilerl/train.py (reward / advantage) | a length term in the reward or a length-aware advantage |
