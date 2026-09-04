# The 600s→1800s timeout fix landed on one of two API paths — 2026-09-05

**Date:** 2026-09-05
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 — the fix is a constant; the 4K
end-to-end confirmation is `pending-remote`
**Task:** local audit, no GPU window (the resident server holds 22.3 of 32 GB)

## Context

`5cdbf7e` raised the server's per-completion wall-clock cap from 600 s to 1800 s, for a
measured reason: **a 4K prefill on sm70 takes ~600 s on its own**, so the 600 s cap fired
before the decode phase began. That commit touched one file, `src/tilerl/server.py`.

There are two front ends over one engine. `/v1/chat/completions` lives in `server.py`;
`/v1/messages` — the API Claude Code speaks — lives in `messages.py` and has its own wait
loop. Only the first was raised.

| path | cap before | cap after this entry |
|---|---:|---:|
| `server.py` `_await_completion` | 1800 s | 1800 s |
| `server.py` `_stream` | 1800 s | 1800 s |
| **`messages.py`** | **600 s** | **1800 s** |

Same engine, same card, same prefill. A 4K+ request through `/v1/messages` raised
`TimeoutError` where the identical request through `/v1/chat/completions` completed.

## What was wrong, and what is not claimed

The fix is one constant, hoisted to `_COMPLETION_TIMEOUT_S = 1800.0` because the 600
appeared **twice** in that function — once as the deadline and once in the error message,
so changing one alone would have made the other lie.

**Not measured here:** that a 4K request through `/v1/messages` now completes. That needs
the card, and the resident 27B server holds 22.3 of 32 GB. What is verified is the
constant, the parity between the two paths, and the gate below. The ~600 s prefill figure
is `5cdbf7e`'s measurement, not a fresh one — it is the reason the fix is correct, and it
was already paid for.

`pending-remote`: one 4K `/v1/messages` request against the live V100, asserting it returns
rather than raising at 600 s.

## The gate

`tests/test_server.py::test_both_api_paths_wait_the_same_wall_clock_for_one_completion`
reads both caps out of the source and asserts they are equal. Reading rather than running,
because the defect is two constants that should be one, and a test that waits 600 s to
observe a timeout would never run in CI.

Mutation control, both directions, both CAUGHT:

| mutation | verdict |
|---|---|
| `messages.py` back to 600 (the bug exactly as found) | CAUGHT |
| `server.py`'s stream cap drifts to 900 | CAUGHT |

The second arm matters: a gate that only checks `messages.py` against a hardcoded 1800
would pass while `server.py` drifted, and the invariant is that they agree — not that
either equals a particular number.

## Rule

**A fix to a duplicated constant is not done until every copy is found.** `5cdbf7e` was
correct and its reasoning was measured; it changed one file because that is where the bug
was observed, and nothing checked whether the same constant lived elsewhere. Grep for the
value before closing a fix like this.

**A second front end over the same engine inherits the engine's costs, not the other front
end's fixes.** `messages.py` was written as a shim with its own loop, which is the right
shape — and it means every timing decision made for `server.py` has to be made twice
unless something asserts they match.

**Prefer reading a constant to exercising it when the failure mode is a wall-clock cap.**
The honest end-to-end test here takes 600 s to fail and would never be in CI, so the
invariant that can be checked cheaply — the two numbers agree — is the one to gate on.
