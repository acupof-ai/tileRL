# The advertised GPU-less path 400'd on its first message, and the suite was green

**Date:** 2026-09-04
**Arch:** cpu (the failing configuration), server path
**Task:** #70

## Context

README tells users without a GPU to run exactly one thing:

> No GPU: `uv run tilerl serve` runs the tiny model on CPU

Send the first message on that path and the server answers **400**:

```
POST /v1/chat/completions -> 400
{"error":{"message":"request (567 tokens) exceeds max_total_tokens (512)"}}
```

Found by the sm90 peer clicking send in a browser, **after** the suite passed and
after the page had been verified for dangling calls. They ran the control before
reporting it — the identical payload against an unmodified checkout returned the
same 400 — so it arrived as a finding, not a suspicion.

## Root cause

`server.py:131` defaulted `max_tokens` to a hardcoded **512**, and the tiny model's
`max_total_tokens` is also **512**. `submit` checks `len(prompt) + max_new_tokens`
against the limit (`engine.py:448-453`), so the sum exceeds it for **any non-empty
prompt**. The failure is not a large prompt or a tight edge case; it is every
request that omits the field.

## Fix

Server-side, one line: the default becomes `min(512, room)` where
`room = max_total_tokens - len(input_ids)`.

The chat page's `sendChat` is *one* client that omits `max_tokens` — fixing the page
would have left every other such client broken, and the server is the side that
knows its own limit. `getattr` chain on `engine.limits` because the test doubles do
not all declare one.

## Why nothing caught it

Two independent blind spots lined up:

**The fixture hid it.** `tests/test_server.py` builds its module engine with
`max_total_tokens=4096` — deliberately, and the comment says why (a one-tool request
is ~1.1k ByteTokenizer tokens). At 4096 a 512 default fits, so the bug is invisible
to every test in the file. The new gate builds a **tight** engine at 512 instead;
that difference is the entire reason this was reachable in production and not in CI.

**The page's request is not testable as code.** `test_every_bare_call_in_the_page_js_resolves`
(from #51) covers dangling *calls*. This is a well-formed call whose *payload* the
server rejects — the same class of gap as the dangling route and dangling button
found earlier the same day: behaviour assembled inside a Python string literal that
`ruff` does not parse and no test drives.

## Verification

- New gate `test_a_request_that_omits_max_tokens_fits_a_tight_context`: 200 and a
  non-null message on a 512-token engine.
- **Negative control**: restoring the hardcoded 512 fails it at
  `request (565 tokens) exceeds max_total_tokens (512)` — the peer's error class,
  reproduced locally. `__pycache__` cleared before the control, per the
  `.pyc`-invalidation trap already recorded in this repo.
- Full cpu suite **252 passed / 8 skipped** (up one), ruff clean.

## Rule

A default that equals a limit is a bug for every input. `512` against a `512`
context has no working case at all, which is why no amount of prompt tuning in a
test would have found it — only a fixture at the real limit, or a browser.

And a test fixture generous enough to make the suite convenient is a fixture that
cannot see the configuration users are told to run. When a doc names a
configuration, something should exercise *that* configuration, not a roomier
relative of it.
