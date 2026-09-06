# /v1/responses, and the SDK that will not tell you a required field is missing — cpu, 2026-09-06

> Status: Shipped; verified live on the V100 (27B NVFP4, sm70) 2026-09-06

## Context

Third API surface, after the chat-completions and messages fixes in the same
day's [entry](2026-09-06-sdk-e2e-found-five-api-deviations.md). Same engine,
same tokenizer, same `render_prompt`, same `split_think`; what differs is the
wire shape — a flat `output` list of typed items instead of `choices`, and
reasoning as its own item rather than a sibling field.

## What Worked

**The schema came off the SDK's pydantic models, not the docs.** `Response`
declares `parallel_tool_calls`, `tool_choice` and `tools` required; each stream
event requires a `sequence_number`. None of those appear in a doc example.
Reading `openai.types.responses` (openai 3.8.0) gave the exact field list, and
`ResponseReasoningItem` in particular requires a `summary` list — we do not
summarise, so the text goes in `content` and `summary` stays `[]` rather than
being faked.

**One vocabulary for three routes.** `_flatten_tools` turns Responses'
`{type, name, parameters}` into the same flat `{name, description,
input_schema}` the chat route's flattener produces, so `render_prompt` renders
and `messages._parse_tool_calls` parses one shape regardless of which API
asked. The route is 250 lines and adds no parser, no template and no second
sampling policy.

**`function_call_output` replay works, and it is what an agent loop needs.**
The second request of a tool loop replays `function_call` and
`function_call_output` items; `_to_messages` renders the first as an assistant
turn carrying `<tool_call>` XML and the second as a `tool_result`, so the
history reaches the same template the first turn used. Gate asserts
`<tool_response>` and the tool's own output are in the prompt the engine
received.

## The finding: a missing required field is invisible to the SDK

Deleting `parallel_tool_calls` from the response body broke **nothing** — 18
passed. The reason, measured rather than assumed:

```
Response.model_validate({id, object, created_at, model, output})  -> ValidationError
Response.construct(**same)                                       -> accepted,
                                                                    parallel_tool_calls = None
```

**The client builds replies with `construct`, not `model_validate`.** So a field
the model declares required arrives as `None` and nothing raises — invisible to
the SDK, visible to any client that reads it. That is the reverse of the usual
failure, where the parser is stricter than the consumer, and it means "the SDK
accepted our reply" is a weaker statement than it looks: it does not check
required fields at all.

The fix is an assertion on the raw JSON, not on the parsed object, because the
parsed object cannot distinguish absent from null. `test_responses_carries_the_
fields_the_model_declares_required` reads the body over httpx and asserts six
fields by name; deleting `parallel_tool_calls` now fails it.

**This corrects a claim I had already written into the module docstring** —
that omitting the field "makes the SDK reject an otherwise correct reply."
It does not. The docstring now says what actually happens and why the tests
carry the check instead.

## Controls

| reverted | goes red |
|---|---|
| `parallel_tool_calls` in the body | `..._fields_the_model_declares_required` (after the fix above; **nothing** before it) |
| `function_call_output` → tool_result | `..._tool_call_and_replay` |
| `blocks_to_text` passing `thinking` through | `..._replaying_a_thinking_block_drops_the_reasoning` |

The third closes the gap 27 named: Claude Code sends the thinking block back in
the next assistant turn, `blocks_to_text` silently drops kinds it does not know,
and that *is* the wanted behaviour — the template re-opens `<think>` only after
the last real user query, so replayed reasoning is off-distribution. It was
untested, which is indistinguishable from a silent format regression. The arm
asserts the reasoning text is absent from the prompt while the same turn's
`tool_use` and the following `tool_result` both still render.

## `scripts/api_e2e.py`, and its own floor

The same checks against a live `--base-url`, for the V100. Two things it does
that the CPU gate cannot: it exercises a real tokenizer (whether `<think>` is
one token decides whether the reply carries a bare closer) and a real model
(whether a `<tool_call>` comes out well-formed at all).

Smoke-tested against the canned server, and the first run **reported "all
checks passed" with 5 of 11 arms skipped** — a deployment answering almost
nothing would have exited 0. It now fails with exit 2 when more than a third of
the arms skip, and names them:

```
6 passed but 5/11 SKIPPED -- too many to call this a pass. Skipped: chat
non-stream, chat reasoning_content (non-stream), chat stream, chat tool_calls,
messages tool round trip
```

## Cost

`scripts/bench_api_routes.py`, n=60, canned engine, so HTTP + render + parse
with no forward:

| route | median | p90 |
|---|---:|---:|
| responses non-stream | 1.01 ms | 1.14 |
| responses stream (drain) | 4.02 ms | 5.05 |
| chat non-stream | 0.90 ms | 1.11 |
| messages non-stream | 3.19 ms | 5.49 |

Responses lands between the two existing routes and nothing else moved outside
the run-to-run spread established in the previous entry (0.96 ms on
`messages non-stream` against deltas of ~0.3). **No claim is made about the
per-request cost of mounting the route**; a real number needs the pod.

## The live run, which the CPU gate could not stand in for

`scripts/api_e2e.py --base-url http://10.37.2.27:8000 --model qwen38-27b`, run by
27 against the V100 on merge sha `33a69a0`: **rc 0, all 11 checks passed, 0
skipped.** The zero matters as much as the eleven — the skip floor exists because
the first canned run reported success with 5 of 11 skipped, so "0 skipped" is what
distinguishes real coverage from a probe that agreed with itself.

Four facts only a real tokenizer and real weights could establish, all of which
SKIP on the canned engine:

| fact | on the V100 |
|---|---|
| is `<think>` one token? | **yes** — reasoning arrives as its own field, 101 chars on the non-stream chat arm, no bare closer in `content` |
| does the 27B emit a well-formed `<tool_call>`? | **yes, on both routes** — `Bash {"command": "ls"}` parsed, and the Messages round trip answered after the `tool_result` |
| does `/v1/messages` return a thinking block? | **yes** |
| does `/v1/responses` return typed items? | **yes** — reasoning + message |

So the five deviations fixed in the previous entry and the route added here are
confirmed against the deployment, not only against a canned reply.

## Not established
- **Stateless only.** `store` is accepted and ignored, so there is no
  `previous_response_id` and no `GET /v1/responses/{id}`. A client that relies
  on server-side conversation state will not work; one that sends its history in
  `input` will.
- **No streaming tool-call deltas on chat completions.** Unchanged from the
  previous entry — Responses streams `function_call_arguments.delta`, chat
  completions still does not.
- **`tool_choice` is echoed, never enforced**, on all three routes. Nothing
  forces or forbids a call.
- **Untested Responses surface**: `include`, `truncation`,
  `max_tool_calls`, image and file inputs, and every hosted tool type
  (`web_search`, `file_search`, `code_interpreter`) — accepted by the request
  model where declared, implemented nowhere.

## Rule

An SDK accepting your response body is not evidence the body is complete: the
OpenAI client builds models with `construct`, so a field its own schema declares
required arrives as `None` without an error. When compliance matters, assert the
field names on the raw JSON — and give any probe that can skip a check a floor
on how much skipping still counts as a pass.
