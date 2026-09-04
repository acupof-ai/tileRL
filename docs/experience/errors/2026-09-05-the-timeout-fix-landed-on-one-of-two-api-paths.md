# The 600s→1800s timeout fix landed on one of two API paths — 2026-09-05

**Date:** 2026-09-05
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 — the fix is a constant; the 4K
end-to-end confirmation landed later the same day (39.1 s, see *Resolved* below)
**Task:** local audit, no GPU window (the resident server holds 22.3 of 32 GB)

## Context

`5cdbf7e` raised the server's per-completion wall-clock cap from 600 s to 1800 s, citing
**"a 4K prefill on sm70 takes ~600 s"** so that the 600 s cap fired before the decode
phase began. That commit touched one file, `src/tilerl/server.py`.

**That figure is withdrawn** — measured 39.1 s for 3478 tokens, 15x lower, and it was a
B=8 whole-tick cost quoted as a single request. See
[2026-09-05-the-600s-that-justified-1800s-was-a-batch-tick.md](2026-09-05-the-600s-that-justified-1800s-was-a-batch-tick.md).
The defect recorded here does not depend on it: whatever the right cap is, the two paths
have to agree on it.

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

**Resolved** (measured after this entry was first written, on the live V100 running
`550740a`, which still carried both 600 s constants): a **3478-token** `/v1/messages`
request **COMPLETED in 39.1 s**, so the cap never fired. The client budget was set to
900 s on purpose — above the server's 600 s — so that whichever side gave up first was
identifiable; a client-side timeout would have said nothing about the server's cap. What
this measurement establishes is that the two paths now agree and that 4K fits comfortably
inside either cap. What it withdraws is the ~600 s premise, in the entry linked above.

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
