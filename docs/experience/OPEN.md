# Open defects

Documented in `errors/` with a fix named and not landed. A PR that lands the fix removes
its line. Writing the entry and adding the line are one act — an entry with `Status: open`
and no line here is the same defect this file exists to stop. Reviewers check a change on the same path against this list.

| entry | path | fix |
|---|---|---|
| [the rollouts grew into the cap](errors/2026-09-06-the-rollouts-grew-into-the-cap.md) | `src/tilerl/eval.py` `MATCHERS`, GRPO advantage | a length term in the reward or a length-aware advantage — `boxed_match` is correctness-only, so nothing prefers the shorter of two correct answers and the policy lengthens until it truncates |
| [a spec rate over a dense roofline](errors/2026-09-06-a-spec-rate-over-a-dense-roofline.md) | a draft step's streamed bytes, nowhere bucketed | extend `scripts/check_scale_f16.py`'s role buckets to the draft path so the speculative ceiling is a number instead of the 32-42% bracket — the four wrong denominators are fixed, but the bracket's width is one unmeasured operand, and 0.85 GB is the head's *resident* size, not what a draft forward streams |
| [a number with no instrument](errors/2026-09-06-a-number-with-no-instrument.md) | five citations of the "~60 µs eager launch floor" | run `scripts/probe_launch_floor.py` on an idle V100 and replace the figure with the measurement, or delete the claim from all five sites — the probe exists and refuses a busy card, and the endpoint holds 28.0 of 32.8 GB |
