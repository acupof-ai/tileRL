# Open defects

Documented in `errors/` with a fix named and not landed. A PR that lands the fix removes
its line. Writing the entry and adding the line are one act — an entry with `Status: open`
and no line here is the same defect this file exists to stop. Reviewers check a change on the same path against this list.

| entry | path | fix |
|---|---|---|
| [the rollouts grew into the cap](errors/2026-09-06-the-rollouts-grew-into-the-cap.md) | `src/tilerl/eval.py` `MATCHERS`, GRPO advantage | a length term in the reward or a length-aware advantage — `boxed_match` is correctness-only, so nothing prefers the shorter of two correct answers and the policy lengthens until it truncates |
| [a spec rate over a dense roofline](errors/2026-09-06-a-spec-rate-over-a-dense-roofline.md) | a draft step's streamed bytes, nowhere bucketed | extend `scripts/check_scale_f16.py`'s role buckets to the draft path so the speculative ceiling is a number instead of the 32-42% bracket — the four wrong denominators are fixed, but the bracket's width is one unmeasured operand, and 0.85 GB is the head's *resident* size, not what a draft forward streams |
| [the NCCL floor has no instrument](errors/2026-09-06-the-nccl-floor-has-no-instrument.md) | ten citations of "21.5 µs NCCL floor" across `docs/design-parallel.md`, `docs/roadmap.md`, `CHANGELOG.md:357` | run `torchrun --nproc_per_node=8 scripts/nccl_probe.py` on the H20 and replace the figure, or mark all ten sites unmeasured — the probe exists and nothing cites it, and the ring-vs-all-gather verdict at `design-parallel.md:191` rests on the number |
| [every Claude Code turn 400s on the second ceiling](errors/2026-09-07-every-claude-code-turn-400s-on-the-second-ceiling.md) | `src/tilerl/messages.py:184`, `:202` | clamp with `engine.room_for(len(input_ids))` — the clamp bounds `max_total_tokens` and not the KV pool, so it lands every request on 32768 exactly and `submit` refuses it by `width - 1`; measured 400 on every `/v1/messages` turn with prompt ≥ 768 and `max_tokens: 32000`, which is every Claude Code turn |
