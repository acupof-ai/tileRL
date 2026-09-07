# Open defects

Documented in `errors/` with a fix named and not landed. A PR that lands the fix removes
its line. Writing the entry and adding the line are one act — an entry with `Status: open`
and no line here is the same defect this file exists to stop. Reviewers check a change on the same path against this list.

| entry | path | fix |
|---|---|---|
| [the rollouts grew into the cap](errors/2026-09-06-the-rollouts-grew-into-the-cap.md) | `src/tilerl/eval.py` `MATCHERS`, GRPO advantage | a length term in the reward or a length-aware advantage — `boxed_match` is correctness-only, so nothing prefers the shorter of two correct answers and the policy lengthens until it truncates |
| [the NCCL floor has no instrument](errors/2026-09-06-the-nccl-floor-has-no-instrument.md) | ten citations of "21.5 µs NCCL floor" across `docs/design-parallel.md`, `docs/roadmap.md`, `CHANGELOG.md:357` | run `torchrun --nproc_per_node=8 scripts/nccl_probe.py` on the H20 and replace the figure, or mark all ten sites unmeasured — the probe exists and nothing cites it, and the ring-vs-all-gather verdict at `design-parallel.md:191` rests on the number |
