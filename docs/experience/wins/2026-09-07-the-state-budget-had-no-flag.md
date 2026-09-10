# The state budget had no flag, so the tiers' own regime was unbenchable — H20, 2026-09-07

> Status: Shipped (flag + gate); the measured cells are pending-remote

## Context

The DRAM and SSD tiers exist for one condition: **concurrent sessions >
the snapshots HBM keeps resident**. Every published verdict on them swept the
budget or the session count, never the ratio, because the numerator was not
reachable from the command line.

`serve`'s flags were `--model --host --port --devices --draft --depth --slots
--blocks --max-ctx --ssd-path --ssd-min-tokens --dram-bytes --max-batch
--no-warmup`. The budget came from one line inside `build_engine`
(`engine.py:1606-1610`): a quarter of free memory after weights and pools, with
no way in.

That default puts the regime out of reach on the box the benches run on. A GDN
snapshot is **157 MiB** measured on the H20, and a quarter of free there is
**17.9 GiB = 116 snapshots**, so DRAM pressure begins near **115 concurrent
agent sessions**. Nothing benchable reaches that. Every tier arm run to date
therefore measured the unpressured side and read `0` promotions — a true number
about the wrong condition.

The V100 is the opposite case and shows the arithmetic is the point, not the
card. Read off the live child (`/health`, pid 3128149, `--max-ctx 32768`, depth 1):
`prefix_state_bytes_budget` **1845067776 = 1.718 GiB**, snapshot 149.6 MiB,
`prefix_entries_capacity` **11** — so 12 sessions is already past the budget there.
Same code, same default, two regimes.

> **Provenance (2026-09-10 cleanup):** the one-off `scripts/bench_h2d.py` was deleted here; rerun it by hand with `TILERL_TARGET=cuda python3 scripts/bench_h2d.py`. The 149.6 MiB size is now derived through precision.nbytes (144 MiB GDN state + 5.6 MiB conv window).

## What Worked

`--state-bytes` end to end: `cli.py:1035` (argparse) → `_build_engine`'s kwarg
(`cli.py:113,142`) → `build_engine`'s kwarg (`engine.py:1532`) → the store.

The precedence is explicit rather than additive, because the CUDA default was
already unconditional:

```python
kw = {}
if state_bytes:
    kw["state_bytes"] = state_bytes
elif backend.device.type == "cuda":
    kw["state_bytes"] = int(torch.cuda.mem_get_info()[0] // 4)
```

An `if state_bytes: kw[...] = state_bytes` placed *after* the cuda branch would
have read as a fix and been one on the cpu target, where the branch never fires —
which is exactly where the check below runs. The `elif` is what makes the flag win
on the card.

Verified on the cpu target, both directions:

| arm | `prefix_state_bytes_budget` |
|---|---:|
| `--state-bytes 1073741824` | 1073741824 |
| no flag | 8589934592 |

`tests/test_server.py::test_serve_state_bytes_reaches_health`, parametrized
`[0, 12345678]`, drives `cmd_serve` and reads `/health`. The control asserts the
default is a **different** value rather than an absent key, since
`prefix_state_bytes_budget` is published either way (`engine.py:856`) and an
`in`-test would pass with the flag ignored.

Negative control run: deleting the two-line pass-through in `cli.py` turns the
`12345678` case **red** (`1 failed, 1 passed`) and leaves the `0` case green —
so the gate fails for the reason under test and not by some second route.

## Rule

**A tier's operating condition needs a flag on both of its terms.** A budget
computed from free memory is not a knob, and a sweep of the other term alone
finds a threshold that reads like a law. Before a tier verdict, check the flag
list actually reaches the regime the verdict is about: the numbers here are
`116` snapshots on the H20 versus `9` on the V100 from the same default.

## Results

| date | commit | machine | target | model | metric | value |
|---|---|---|---|---|---|---:|
| 2026-09-07 | pending | mac (cpu) | cpu | tiny | `--state-bytes 1 GiB` honoured | 1073741824 |
| 2026-09-07 | pending | mac (cpu) | cpu | tiny | default budget, same run | 8589934592 |
| 2026-09-07 | pending | H20 | cuda | qwen38-27b | snapshots at the default budget (derived) | 116 |
| 2026-09-07 | 1a3930a | V100 | cuda | qwen38-27b | `prefix_entries_capacity`, live child (read) | 11 |

The H20 row is **derived**, not timed: `mem_get_info()[0] // 4` divided by the
measured 156.9 MiB snapshot. It prices the regime; it is not a wall clock. The V100
row is **read** from the live child's `/health` — an earlier draft derived it as 9
from 8 GiB at 144 MiB and both operands were wrong (8 GiB is `PrefixStore`'s own
default, not the quarter-of-free this card gets after pools; 144 MiB is the bf16
snapshot, not this checkpoint's 149.6). Two wrong operands composed into a
plausible number, which is why the row now cites a reading.

Raw artifacts: `tests/test_server.py::test_serve_state_bytes_reaches_health`.
