# `--dram-bytes` on `tilerl serve` — cpu, 2026-09-06

> Status: Shipped (flag + gate). The wall clock at 2/8/12 sessions is
> pending-remote — this entry ships reachability, not a perf claim.

## Context

The host snapshot tier was reachable only by editing `build_engine`'s default. The
condition that makes it win —
[`sessions > HBM snapshot budget`](../errors/2026-09-05-a-two-variable-condition-read-as-a-dead-end.md)
— is a property of the deployment, so an operator who cannot set it from the command
line has to patch source to run the multi-session case at all. The same entry recorded
why it stays default-off: at one session the tier is **1.51x worse** on wall clock, 43
demotions and 0 promotions.

## What Worked

`--dram-bytes` on `tilerl serve`, default 0, forwarded through `_build_engine`. Two
things had to change beyond the flag itself.

**The tier was nested under `if backend.device.type == "cuda"`.** With it there the flag
is inert on the CPU target, which is the only target CI runs — the gate would have been
green with the tier never constructed. It is now a sibling of the `state_bytes` branch,
for the reason the SSD tier already is: host-to-host is a real demote and promote, and
the CPU target is where that is checked.

**`/health` published the wrong operand.** `dram_bytes` is `self._used`, bytes held, so
it reads 0 both when the tier is off and when it is on with nothing demoted yet. Added
`dram_budget = self.budget_bytes`.

The gate drives `cmd_serve` end to end with `uvicorn.run` monkeypatched to hit `/health`
in process, two arms: `--dram-bytes 12345678` asserts `dram_budget == 12345678`, and the
control asserts the key is **absent** — not 0, because 0 is what the tier reports when it
is on.

Four negative controls, each breaking one hop, each printing the assertion that fired:

| broken | assertion that fired |
|---|---|
| `_build_engine` drops the forward | `dram_* served: {}` |
| tier re-nested under `cuda` | `dram_* served: {}` |
| publish only the old `dram_bytes` | `dram_* served: {…7 keys, dram_bytes: 0}` |
| tier always constructed | **control arm**: `the tier is on without the flag: dram_budget=4294967296` |

The third is the one that carries information: the tier is on, all seven `dram_*` keys
are present, and `dram_bytes` is still 0 — identical to the off arm. An equality on it
would have passed with the tier dead. The fourth fails the other arm, so the two arms
have different killers.

## Rule

When a flag has to cross several hops to matter, assert the value at the far end and pick
a field that differs between on and off. A counter that starts at 0 reads the same as an
absent feature, so a gate keyed on it passes for the wrong reason — and each arm of a
two-arm gate needs a control that turns *that* arm red.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-06 | pending | mac | cpu | tiny | n/a | n/a | n/a |

No perf surface: the flag defaults to 0 and `stats()` gains one dict key read once per
`/health`. `343 passed, 14 skipped` on cpu, ruff clean.

Still unmeasured, and the reason this is not a perf entry: the V100 wall clock per turn
at 2 / 8 / 12 concurrent sessions, tier off against on at the 9-snapshot-equivalent
budget. The 0-vs-24 promotion counts in the errors entry are hit counts, not time.
