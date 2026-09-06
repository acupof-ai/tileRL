# Open defects

Documented in `errors/` with a fix named and not landed. A PR that lands the fix removes
its line. Writing the entry and adding the line are one act — an entry with `Status: open`
and no line here is the same defect this file exists to stop. Reviewers check a change on the same path against this list.

| entry | path | fix |
|---|---|---|
| [the rollouts grew into the cap](errors/2026-09-06-the-rollouts-grew-into-the-cap.md) | `src/tilerl/eval.py` `MATCHERS`, GRPO advantage | a length term in the reward or a length-aware advantage — `boxed_match` is correctness-only, so nothing prefers the shorter of two correct answers and the policy lengthens until it truncates |
| [the save stage had no timer](errors/2026-09-06-the-save-stage-had-no-timer.md) | `max_pending=32` at `src/tilerl/kv_cache.py:405`, and the pod's write rate | run `scripts/probe_save_ms.py` on the pod's `/work` from inside the container namespace, then size `max_pending` against the measured per-save cost — the five `~100 ms` comments are corrected to the 641.8 ms measured here, but the cap still holds 32 × 320.6 MiB = 10.0 GiB of a 31 GB host and was set against a figure 6.4x low |
