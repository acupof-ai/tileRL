# Three API surfaces matched to spec, and the SDKs found what hand-written JSON could not — cpu, 2026-09-06

> Status: Shipped (chat completions + messages); V100 numbers `pending-remote`

## Context

The two live routes were gated by hand-written asserts on response dicts. That
gates the fields you thought to check. `#155` asked for the other direction:
fixtures produced by the server, parsed by the client that actually consumes
them. `uv add --dev openai anthropic` (openai 3.8.0, anthropic 1.4.0), one
uvicorn over the canned engine, and the official SDKs driving both routes.

## What Worked

**Five deviations, none of which the existing 34 server tests could see.** Each
was confirmed in the source, not just by a failing assertion:

| # | Deviation | Evidence on main |
|---|---|---|
| 1 | chat non-stream dropped the reasoning the stream returns | `server.py:217` called `strip_think` while `_stream` used `split_think` |
| 2 | chat `tools` silently ignored | 0 occurrences of `tools` in `server.py`; raw `<tool_call>` XML in `message.content`, `tool_calls: None` |
| 3 | `/v1/messages` `thinking` returned no thinking block | 0 occurrences of `"type": "thinking"` in `messages.py` |
| 4 | chat rejected a bad field with FastAPI's 422 `detail` | 0 `RequestValidationError` handlers |
| 5 | `/v1/messages` likewise | same handler, same absence |

**Defect 1 was introduced by #159 three hours earlier**, which put
`reasoning_content` on the streaming path only — so flipping `stream` changed
which fields a reply has. That is exactly the class an SDK catches and a dict
assert does not: both paths individually looked correct.

**One handler fixed 4 and 5 together**, choosing the envelope by request path
since one app serves both APIs. The evidence it was shared: after writing it,
the *Anthropic* arm went `XPASS(strict)` without that route being touched.

**Defect 2 needed no new parser.** `messages.py` already had
`_parse_tool_calls`, and `render_prompt` already accepted `tools`; the chat
route simply never passed or read them. The only new code is `_flatten_tools`,
12 lines turning OpenAI's `{type, function: {...}}` into the flat
`{name, description, input_schema}` both the template and the existing parser
want — so a call means the same thing whichever API asked for it.

**The SSE loop could not express a third block type.** Adding the thinking
block turned `messages.py`'s `tool`/else pair into a `KeyError` *inside the
generator*, past the 200 header, so the client saw `RemoteProtocolError:
incomplete chunked read` rather than an error — the same shape the module's own
ponytail note warns about. Now dispatched per type.

## The controls: every fix has one, and one of them was missing

Reverting each hunk individually, with `__pycache__` cleared each time:

| reverted | goes red |
|---|---|
| the validation handler | both envelope tests (chat **and** messages) |
| `reasoning_content` on the non-stream reply | `..._same_field_on_both_paths` |
| `_parse_tool_calls` on the response | `..._tools_come_back_as_tool_calls` |
| the thinking block | `..._thinking_is_a_thinking_block` |
| **`_flatten_tools` at the render site** | **nothing — 12 passed** |

That last row is the finding. Deleting the code that renders the schemas into
the prompt broke no test, because the canned engine emits the tool call whether
or not the prompt ever defined the tool. Parsing the reply cannot tell you the
model was told what `Bash` is. `test_chat_tools_reach_the_prompt` now asserts
`<tools>` and `"name": "Bash"` in the prompt the engine received, and that
control goes red.

**`xfail(strict=True)` carried the red state across tranches**, and it is also
verified rather than assumed: wiring `reasoning_content` in while its marker
stood produced `[XPASS(strict)]` and failed the run, so a fix cannot land
without deleting its marker.

## Cost

`scripts/bench_api_routes.py`, n=60 per row, canned engine, so this is
HTTP + render + parse with no forward in it. Same script on both trees:

| route | main (d2a0953) | this branch | delta |
|---|---:|---:|---:|
| chat non-stream | 1.02 ms | 0.87 ms | −0.15 |
| chat non-stream + tools | 0.99 ms | 0.94 ms | −0.05 |
| chat stream (drain) | 54.95 ms | 57.32 ms | +2.37 |
| messages non-stream | 3.34 ms | 3.65 ms | +0.31 |
| messages non-stream + thinking | — | 4.06 ms | new arm |
| messages stream (drain) | 4.39 ms | 5.05 ms | +0.66 |

**None of these deltas is a measurement of my change.** Re-running the same
branch three more times, the medians move by more than the deltas do:

| route | branch, 4 runs | spread | delta claimed above |
|---|---|---:|---:|
| chat non-stream | 0.87 / 0.96 / 0.95 / 0.93 | 0.09 ms | −0.15 |
| messages non-stream | 3.65 / 4.51 / 4.61 / 4.16 | 0.96 ms | +0.31 |

The `messages` spread is 3x its own delta, and `chat non-stream` reads *faster*
after adding work to it — both say run-to-run variance dominates at n=60. The
stream rows are worse: that loop sleeps 20 ms per iteration
(`server.py:370`), so a canned three-peek reply spends tens of ms asleep and
the parse is a rounding error inside it. The honest statement is that no row
moved outside its own spread, and this bench cannot resolve a per-request cost
of this size. Sizing the real per-token cost needs the V100:
`pending-remote`.

## Not established

- **No `/v1/responses`.** The route 404s; the OpenAI SDK's `responses.create`
  is the tranche-(c) target and nothing here implements it.
- **`tool_choice` is accepted and ignored.** It is a field on the request model
  so a client sending it is not rejected; nothing forces or forbids a call.
- **The chat stream does not emit `tool_calls` deltas.** A streaming client
  asking for tools gets the XML in `content` deltas. Non-stream is fixed;
  streaming tool calls are not, and no test covers them.
- **`signature` on the thinking block is `""`.** We sign nothing; the field
  exists for replaying a block to the real API, which this is not.
- **No `ping` event** on the Anthropic stream. The SDK tolerates its absence,
  which is why this is a note and not a fix.
- **Canned replies, not a model.** Every number and every parse here ran
  against `_ScriptedEngine`. The V100 has served the reasoning split (27
  verified it on 9927308) but not the tool or thinking paths.

## Rule

A route is only as compliant as the client that parses it. Gate an API by
driving the official SDK against a server-produced fixture — and when the
fixture is canned, add one assertion on what the route *sent*, because a
canned reply makes the request side untestable and every reply-side assertion
still passes.
