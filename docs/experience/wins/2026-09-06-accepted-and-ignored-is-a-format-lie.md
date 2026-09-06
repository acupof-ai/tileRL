# A field accepted and ignored is a format lie, and three of them were never even received — cpu, 2026-09-06

> Status: Shipped (refusals + gate); `stop`/`stop_sequences` were tracked open and
> are now implemented — [wins/2026-09-06-stop-sequences-matched-on-text-not-token-ids.md](2026-09-06-stop-sequences-matched-on-text-not-token-ids.md)

## Context

The two entries before this one shipped three API surfaces and left a list of
fields "accepted where declared but implemented nowhere." 27's rule closed it:
a field we take and ignore is a lie with a delayed cost, because the client's
**next** request is built assuming the first one honoured it. `store=true`
means the next request may carry only `previous_response_id`; a hosted tool type
means the client expects the *server* to run the search. Answering as if we had
is worse than refusing.

## What Worked

**One helper, three routes, no new error path.** `prompt.refuse_unsupported(**fields)`
raises `ValueError` on any field whose value is truthy; all three routes already
convert `ValueError` into their own 400 envelope, which is the same mechanism
`blocks_to_text` uses to reject an image block rather than send the model a turn
missing its subject. The caller passes only what it cannot implement, so the
refusal is a list at each call site instead of a policy hidden in a helper.

Refused, one parametrised SDK arm per field:

| field | why it cannot be honoured |
|---|---|
| `store=true` | no server-side state, so the next turn's `previous_response_id` would 404 |
| `previous_response_id` | same, from the other end |
| `include` | asks for fields we never produce (`reasoning.encrypted_content`) |
| `truncation="auto"` | asks the server to drop old turns; we never do |
| `tool_choice` beyond auto/none | we render tools into the prompt and cannot force or forbid a call |
| hosted tool types | `web_search`, `file_search`, `code_interpreter`, `computer`, `image_generation`, `mcp`, `local_shell`, `custom`, `apply_patch`, `shell` — the provider's to run |
| `context_management` | **only the edits we do not perform** — see below |

Still accepted, and a test asserts the refusal is **not** blanket:
`truncation="disabled"` (that is our behaviour), `tool_choice` auto/none (a hint
we honour by rendering or omitting the tools), `metadata` (echoed back),
`parallel_tool_calls` (we never emit parallel calls, so either value is
truthful).

## The finding: three of them were not accepted-and-ignored, they were discarded

`previous_response_id`, `include` and `truncation` **could not be refused at
all**, because they were never declared on the request model. pydantic drops an
undeclared field without a trace:

```
ResponsesRequest.model_validate({..., "previous_response_id": "resp_1"})
  .model_extra  ->  None
```

So the value never reached the handler. "We accept and ignore it" was doubly
wrong: we did not receive it, and the client got a 200 with no sign its field
had vanished — strictly worse than ignoring, because there is nothing in the
code to grep for. Each had to be **added** to the model in order to be
rejected, which is the opposite of the usual fix direction.

The same was true of `stop` on chat completions and Responses: not a field,
silently dropped. Both now declare it solely to refuse it.

## Refusing a whole field broke the live client, and the gate caught it

I wrote "no client is known to send any refused field" into this entry, and the
rollout gate failed on the next full run: `test_rollout_records_carry_the_episode_
tag` drives the real `claude` CLI against our server, and it went from passing on
main to failing on the branch, twice each.

**Claude Code sends `context_management` on every request.** Captured by logging
the CLI's own body through a middleware rather than reasoning about it:

```
keys: context_management, max_tokens, messages, metadata, model,
      output_config, stream, system, thinking, tools
context_management = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
```

And the edit asks for behaviour **we already have**: `blocks_to_text` drops
thinking blocks when replaying history, so `clear_thinking` is satisfied by
construction. Refusing it was the mirror of the defect this tranche exists to
fix — a 400 for a field we do honour, instead of a 200 for one we do not.

`docs/lessons/messages_shim_stage1.md:25` had listed `context_management` among
Claude Code's top-level keys since 2026-09-02. I read that file for the field
list and did not connect it to what I was about to refuse.

The fix refuses **per edit**, not per field: `_unsatisfied_edits` returns only
the edit types outside `_SATISFIED_EDITS` and names them, so
`clear_tool_uses_20250919` is refused as `context_management edits
['clear_tool_uses_20250919']` while `clear_thinking` passes. Two arms now: one
that the unsatisfied edit is refused, one that the edit Claude Code always sends
is accepted.

A side effect worth recording, because it cost three more red arms: my first
attempt let `context_management` name its own edit by having a truthy **string**
value replace the kwarg name. That immediately made
`previous_response_id="resp_1"` report "resp_1 is not supported" — naming the
value instead of the field. A convenience keyed on value *type* reaches every
caller that happens to pass that type, so the two intents are now separate
parameters: `refuse_unsupported(*already_decided, name=truthy)`. Positional
strings are the exact text to name, keyword args name themselves and never their
value. Verified: `previous_response_id="resp_secret"` cannot put the value in the
400, and the positional form still names `clear_tool_uses_20250919`.

**The gate this earned.** `test_no_refused_field_is_sent_by_the_real_client`
drives the CLI and asserts, against captured bodies, that no refused field is
sent and that every `context_management` edit the client asks for is one we
satisfy. `rollout.capture_bodies(app)` is the reusable instrument — the V100
request log records token ids, not bodies, so a wire-shape question has no other
answer. Control: restoring the whole-field refusal turns that test red on its own
assertion ("the CLI asks for an edit we refuse"), alongside the episode test.

## Controls

| reverted | goes red |
|---|---|
| `refuse_unsupported` never raises (`named = []`) | all 9 refusal arms |
| `previous_response_id` undeclared again | its own arm, with `AttributeError` — the silent-drop path |
| `refuse_unsupported` refuses everything (`named = list(fields)`) | 4 ordinary arms, including plain `chat non-stream` |

The third is the one worth keeping: a blanket refusal passes every test that
asserts a 400 and breaks the requests that should work, so "the refusals fire"
and "the refusals are narrow" are two claims needing two controls.

## `stop_sequences` is a different kind of field, and was tracked open

**Superseded 2026-09-06 (same day): the field is implemented, not refused** —
matched on decoded text, because the guess below that the token-level mechanism
was nearly enough is wrong (a stop string need not start at a token boundary).
[wins/2026-09-06-stop-sequences-matched-on-text-not-token-ids.md](2026-09-06-stop-sequences-matched-on-text-not-token-ids.md).
The paragraph is kept because the reasoning it contains is the mistake.

Refusing it was right that day, but it is a **core field of both published APIs**,
not an unsupported extension, so it goes to `OPEN.md` rather than onto the
hosted-tools list. `messages.py:244` already carried a comment admitting the
field was ignored, which is the defect describing itself and being left alone.

The engine is closer than the refusal implies: `SamplingParams.stop_token_ids`
exists and **is honoured** at `engine.py:1129`, but `prompt.py:57` fills it from
the tokenizer's eos ids and from nothing else, so no request field reaches it —
and it stops on a *token*, while a stop sequence is a string that may span
tokens and must be truncated out of the reply. Details and the named fix:
[errors/2026-09-06-stop-sequences-accepted-and-never-applied.md](../errors/2026-09-06-stop-sequences-accepted-and-never-applied.md).

## Which client sends what, from the captures

| client | instrument | sends |
|---|---|---|
| Claude Code CLI | `capture_bodies` over the rollout gate | `context_management`, `output_config`, `thinking`, `tools`, `system`, `metadata`, `stream`, `max_tokens`, `messages`, `model` — and no refused field |
| openai / anthropic SDKs | `tests/test_api_sdk.py` | the OpenAI-only fields, including every refused one, asserted 400 per field |
| tilerl chat page | `tests/test_chat_ui.py` | none of the refused fields |

`stop` and `stop_sequences` appear in **neither** capture, which is why refusing
them broke nothing that day — a measurement rather than the inference in the first
draft of this entry. It is also why implementing them the same day changed no
observed client's behaviour.

**What the capture does not prove.** It records what the CLI sends *on the task
shape the gate runs*. `output_config` and `thinking` already vary between requests
in one episode, so a different prompt can take a path that sends more. The gate
establishes "no refused field on the exercised path", not "no refused field ever",
and reading it as the latter is the same over-claim this entry was written to
correct.

## Not established

- **No live run.** Canned engine only; the refusals are wire-level and the pod
  cannot change them, but `api_e2e.py` has no refusal arms yet.
- **One refused field DID break a live client, and the claim that none would
  was wrong.** See the section above; the corrected statement is that the chat
  page sends none of the refused fields and Claude Code sends
  `context_management` on every request. Nothing else is known to be sent, which
  is still an argument from the requests we happen to have recorded rather than a
  measurement of what clients send.
- **`tool_choice: {"type": "function", "name": X}` could plausibly be honoured**
  by rendering only that tool, and is refused rather than attempted. Not
  investigated.

## Rule

Before deciding a field is "accepted and ignored," check that it is declared —
pydantic discards what it does not know, so the value may never have arrived,
and there is nothing in the code to find. Then refuse what you cannot honour,
and write **two** controls: one that the refusal fires, one that it is narrow.
