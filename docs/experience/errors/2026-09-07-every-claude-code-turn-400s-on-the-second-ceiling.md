# Every Claude Code turn 400s on the second ceiling

`Status: fixed` — `messages.py` clamps with `engine.room_for()`.

## Context

The agent trial: drive the V100 endpoint with a real Claude Code session and
measure tool round trips and wall clock per turn. The recipe, as agreed:

> Build a scratch repo with a one-line bug — `calc.py` returning `a - b`,
> `check.py` asserting `calc.add(2, 3) == 5` — confirm `python3 check.py` fails
> with `AssertionError`, then point a Claude Code session at the endpoint with
> `ANTHROPIC_BASE_URL`, `--permission-mode acceptEdits`, and the prompt "Read
> calc.py, find why check.py fails, fix calc.py, then run `python3 check.py`
> and tell me the output." Answer three questions: does a session complete at
> all, how many tool round trips, wall clock per turn. Do not report a
> question as clean when no session completed.

Endpoint stamp read from `/proc/2807036/cwd` at trial start **and again at
trial end**: `26db7bd` both times, so no redeploy landed mid-trial. Both
`1a0e49b` (#201) and `43881e9` (#203) confirmed ancestors. Flags:
`--max-batch 1 --max-ctx 32768 --depth 1`, `blocks_total: 2048`.

The session produced three lines and no turn:

```
[claude-code:unrecognized_model] {"model":"qwen38-27b","query_source":"sdk"}
API Error: 400 request (32768 tokens) exceeds KV pool capacity
```

Question 1 is answered: **no session completes**. Questions 2 and 3 are
unanswerable until this is fixed, and per the recipe's own rule they are not
reported as clean.

## Root cause

`messages.py:184` reads one of the two ceilings `submit` enforces:

```python
engine_limit = getattr(getattr(engine, "limits", None), "max_total_tokens", 0)
...
budget = max(1, engine_limit - len(input_ids)) if engine_limit else req.max_tokens
```

`submit` enforces two (`engine.py:517` and `:523`): `max_total_tokens`, and the
KV pool including the `width - 1` drafts a verify tick materializes past the
last token. The clamp bounds the first and not the second, so it lands the
request exactly on the pool's edge:

- `budget = 32768 - prompt`, so for any `prompt >= 768` the client's
  `max_tokens: 32000` no longer binds and `total = prompt + budget = 32768`
  **exactly**, whatever the prompt length.
- `blocks_for_tokens(32768 + width - 1) = 2049 > 2048`. Refused by `width - 1`
  tokens.

Claude Code always asks `max_tokens: 32000` and its system prompt alone is
thousands of tokens, so every turn lands in the failing region.

`Engine.room_for()` (`engine.py:485`) returns exactly `min` of both ceilings
and exists for this. The chat and Responses routes adopted it in #195;
`messages.py` did not, because the review question at the time was the
*omitted* cap — Anthropic requires `max_tokens`, so there was no omitted case
on this route. The **clamp** path needed it too, and that was not asked.

## Measured

Live, against `26db7bd`. Prompt 1212 tokens, so `room_for(1212) = 32768 - 1212
- 2 + 1 = 31555`:

| route | prompt | max_tokens | result |
|---|---|---|---|
| `/v1/messages` | 1212 | 32000 | 400 `exceeds KV pool capacity` |
| `/v1/messages` | 1212 | 64 | ok |
| `/v1/messages` | 412 | 32000 | ok (client's cap binds) |
| `/v1/messages` | 1212 | **31555** | ok |
| `/v1/messages` | 1212 | **31556** | 400 |
| `/v1/chat/completions` | 1212 | 32000 | 400 `exceeds max_total_tokens` |
| `/v1/chat/completions` | 1212 | omitted | ok |

Two things the boundary pins that reading the code does not. `room_for`'s value
is tight to the token — 31555 passes, 31556 refuses — and the clamp computes
31556, off by exactly `width - 1`. And the failure itself proves `width >= 2`
from outside the process: at width 1, 32768 tokens is exactly 2048 blocks and
the arm would have passed.

The chat route is correct on both arms and is the control: it refuses a
client-named cap that does not fit (OpenAI semantics) and calls `room_for` when
the cap is omitted. The defect is one route wide.

Recorder rows: **still 36, no row for the failed attempt.** The 400 is raised
inside `engine.submit`, before `_record` runs, so the recorder cannot see this
class of failure at all. Row 41 from an earlier turn shows the arithmetic
plainly — `budget: 32457`, `engine_limit: 32768`, `prompt_len: 311` — the
`by_pool` term appears nowhere in the row because it appears nowhere in the
clamp.

## Fix

`messages.py` clamps with `engine.room_for(len(input_ids))` instead of re-deriving
one ceiling. `max(1, ...)` on the result is load-bearing: `room_for` returns 0 when
the prompt alone does not fit, and `submit` skips both ceiling checks when
`max_new_tokens` is 0. `engine_limit` stays, read for the recorder row only — a row
whose `budget` is below `engine_limit - prompt_len` is one the pool capped, which the
old row could not express.

The gate is a real 32-block engine (512 pool tokens against a 4096 context, so the
pool binds by ~9x) that asserts it binds on the pool before asserting anything else;
a fake with a declared pool and width would restate `room_for`'s formula and let a
broken `room_for` pass. The boundary is asserted at the seam, not through the route:
after the fix a client cap above `room_for` is clamped *down* rather than refused.

Negative control, hand clamp restored, red on its own assertion:

```
AssertionError: a 32000-token ask 400-ed on a pool-bound engine:
{"message":"request (4096 tokens) exceeds KV pool capacity"} — the clamp bounds
one of submit's two ceilings
assert 400 == 200
```

`total = 4096` exactly, the same landing-on-the-edge shape as the V100's 32768.

## Rule

**A caller that re-derives a guard's arithmetic reproduces the subset it
remembers.** Three instances now on the same seam, each one route over:
`limits` missing from `DataParallelEngine` (400 on every turn under
`--devices`), `room_for` missing (500 on every omitted cap), and now a route
that has `room_for` available and re-derives half of it by hand. The gate at
`tests/test_server.py:978` enumerates what the routes *call* and checks every
engine implements it; it cannot see a route that implements the arithmetic
itself. A guard's arithmetic lives in one place and callers ask it, or each
caller carries its own subset of the ceilings.

**A cap the client always sends still needs the clamp path reviewed.** The
review that adopted `room_for` asked "which routes have an omitted-cap case",
which is the wrong question by one step: `messages.py` has no omitted case and
still needed the same arithmetic. Ask what the route does with the value, not
whether the value is present.

## Second defect, found by #201's instrument on the live endpoint

`unknown_fields` produced its **first non-null value in production** on this endpoint:
`{'tool_choice': 'dict{type}'}`. `tool_choice` is a documented Anthropic parameter that
`MessagesRequest` declared nowhere, so `extra="allow"` kept it and nothing read it — the
request was answered as if the forced tool had been honoured.

`/v1/responses` has refused it since it landed (`_unsupported_choice`, now
`prompt.unsupported_choice`): we render tools into the prompt as text and cannot force or
forbid a call, so anything stronger than a hint is unimplementable. `/v1/messages` never
got the treatment. `refuse_unsupported`'s own docstring names the cost — "the client's NEXT
request assumes the first one applied it, so the lie surfaces a turn later" — which for
`tool_choice` means the client believes the tool it forced was the tool that ran.

Fixed by declaring the field and refusing beyond auto/none, reusing the Responses helper
rather than writing a second one; it takes either spelling, since Anthropic's `any`/`tool`
and OpenAI's `required`/`{type: function}` all force a call. Six arms, and the three
permissive ones matter as much as the three refusing: a fix that refused every
`tool_choice` would pass a one-sided test, and `auto` is the value a client sends when it
means "your choice" — refusing it breaks a request that asked for nothing. Control with the
refusal removed: 3 red, 3 green, each red naming its own value.

**The rule:** the instrument found this, not a reading. `unknown_fields` had been null on
every synthetic request and went non-null the first time a real client-shaped body reached
it — a declared-field list is only as good as the bodies it has actually seen.

**And then the instrument's own limit, measured.** `unknown_fields` reports what a request
*carried*, so it finds one field per client that happens to send it —
`tool_choice` surfaced that way on a route live for days. So the set was enumerated instead
of discovered: `scripts/capture_client_body.py` drives the real CLI against a recording
stub and diffs every field sent against `MessagesRequest.model_fields`. **Result: the CLI
sends 10 fields, we declare 14, and 0 are undeclared** — `context_management, max_tokens,
messages, metadata, model, output_config, stream, system, thinking, tools`. Negative
control: hiding `thinking` and `tools` from the declared set makes the diff name both, so
the zero is a measurement and not an empty diff.

That zero corrects a claim I had written from the same evidence that produced the fix:
`tool_choice` is **not** sent on every turn — it is in neither captured request. It is
conditional, which is exactly why it evaded every earlier observation. The refusal is still
right (a forced call is unimplementable here), and the reason the permissive arms matter is
the one above, not a frequency claim I had not measured.
