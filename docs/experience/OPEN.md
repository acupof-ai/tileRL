# Open defects

Documented in `errors/` with a fix named and not landed. A PR that lands the fix removes
its line. Writing the entry and adding the line are one act — an entry with `Status: open`
and no line here is the same defect this file exists to stop. Reviewers check a change on the same path against this list.

| entry | path | fix |
|---|---|---|
| [the rollouts grew into the cap](errors/2026-09-06-the-rollouts-grew-into-the-cap.md) | `src/tilerl/eval.py` `MATCHERS`, GRPO advantage | a length term in the reward or a length-aware advantage — `boxed_match` is correctness-only, so nothing prefers the shorter of two correct answers and the policy lengthens until it truncates |
| [a miss self-reinforces](errors/2026-09-07-a-miss-self-reinforces.md) | `src/tilerl/engine.py:1031` `_finish_prefills` | one 31k-token miss publishes 62 chunk entries into a 6-snapshot budget and evicts every other session's shared head, so 11 of 12 sessions miss in sequence at 14.1 s each — the decode publish site was fixed by REPLACE, the prefill site was not, and `prefill_chunks` alone exceeds a pressured card's budget. **Cross-session half fixed** by `len(tokens) × sharers` eviction (+42% reuse, [wins/2026-09-08](wins/2026-09-08-evict-by-length-times-sharers.md)); the single-conversation half is untouched — every entry has `sharers == 1` before any match, so the shortest, which is the head, still goes first. Its fix is the publish count, `xfail(strict)` in `tests/test_kv.py::test_a_prompts_own_publishes_evict_the_prefix_it_shares` |
