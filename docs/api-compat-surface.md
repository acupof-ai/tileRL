# HTTP API compatibility surface

Three completion routes share one engine and present three vendor shapes:

- `POST /v1/chat/completions` — OpenAI Chat Completions (streaming SSE and JSON).
- `POST /v1/messages` — Anthropic Messages.
- `POST /v1/responses` — OpenAI Responses.
- `WS /ws/chat` — the browser playground's own transport (not an OpenAI surface;
  see `web/README.md`).

This page records where the OpenAI-flavoured routes match the published schema and
the deliberate, named deviations, so a client integration does not discover them by
crashing. Verified against the live V100 serve on 2026-09-16 with `openai-python`
2.15.0 (short and 3 KB tool arguments) and raw `curl`. It is descriptive of the
implemented surface; the source of truth remains the routes in `src/tilerl/`.

## What matches OpenAI

- **Streaming chunk envelope** (`src/tilerl/server.py`, `_chat_chunk`): `id`,
  `object: "chat.completion.chunk"`, `created`, `model`,
  `choices[0].{index, delta, logprobs, finish_reason}`, `usage`,
  `system_fingerprint`; `[DONE]` terminator.
- **Non-streaming object**: `object: "chat.completion"`,
  `choices[0].message.{role, content, tool_calls}`, `usage`, finish reason.
- **`stream_options.include_usage`**: every content chunk carries a cumulative
  `usage` (vLLM-style continuous stats) and a final `choices: []` usage-only chunk
  is emitted, matching the OpenAI terminal-usage convention.
- **`finish_reason`**: `stop` / `length` / `tool_calls`; an explicit stop sequence
  ends as `stop` and takes precedence over the token cap.
- **4xx error envelope**: `{"error": {"message", "type", "param", "code"}}`
  (e.g. `max_tokens: 0` → 400 `invalid_request_error` with `param`/`code: null`).
- **Tool calls** (non-streaming) and the streaming `tool_calls` delta both carry
  `index`, `id`, `type: "function"`, `function.{name, arguments}`.

## Deliberate deviations

### 1. `delta.reasoning_content` is a vLLM/DeepSeek extension, not OpenAI

Streaming reasoning arrives as a top-level `delta.reasoning_content` string (the
vLLM/DeepSeek convention). OpenAI's own Chat Completions schema has no such field.
`openai-python` keeps unknown delta fields (they do not raise), but a client that
only reads official fields ignores the thinking. The Anthropic route exposes the
same content natively as a `thinking` content block. Present by design.

### 2. Streaming tool-call arguments are one complete frame, not token shards

OpenAI streams a tool call as several deltas: a first frame that establishes
`index`/`id`/`name` (with empty `arguments`), then incremental `arguments`
fragments the client concatenates. This server emits **one** `tool_calls` delta
per call whose `function.arguments` is the **complete JSON string**; long
arguments are not split.

Verified with `openai-python` 2.15.0 against the live server: both a short call
(`{"city":"Paris"}`) and a 3049-character arguments object arrived as a single
frame and parsed to valid JSON with `id`/`name` intact. Because the SDK's
accumulator concatenates, a single full frame is the degenerate (correct) case,
so the official SDK does not break. A hand-rolled client that *expects* a leading
name-only frame before any arguments will not see one — treat the first
`tool_calls` delta as self-contained. The trade-off is no per-character tool-arg
"typewriter" streaming. YAGNI: kept whole until a client needs the shards.

### 3. In-band streaming error `type` values are a small internal set

A mid-stream failure (HTTP 200 headers already sent) is delivered as
`data: {"error": {"message", "type"}}` followed by `[DONE]`, which is the same
envelope placement OpenAI uses, but the `type` is one of `api_error` /
`internal_error` rather than OpenAI's full taxonomy. The message is preserved and
`openai-python` raises an `APIError`, so no content is lost; only programmatic
classification by exact OpenAI type strings differs.

### 4. Capacity refusal is HTTP 503 `overloaded_error`, deliberately not 429

When the engine is at its in-flight cap, submit returns **503** with
`{"error": {"type": "overloaded_error", "message", "inflight", "cap"}}`
(`overloaded_body`). This is the Anthropic overloaded value, not OpenAI's
`429 rate_limit_exceeded`; there is intentionally **no `Retry-After`** — capacity
is the server's responsibility, and the machine-readable `inflight`/`cap` fields
let a caller decide whether to queue. Consequence: `openai-python` raises a
server/`APIStatusError` and does **not** apply its built-in rate-limit retry; a
client that wants backoff must read this response itself.

### 5. Pure-reasoning truncation returns `content: ""`, not `null`

When a turn spends its entire token budget inside reasoning, the non-streaming
`message.content` is an empty string while `reasoning_content` is populated (a
tool-call-only turn correctly returns `content: null`). This is valid for the
SDK, but a UI may want to treat empty `content` as "no answer" rather than render
an empty bubble. Tracked as a possible one-line self-consistency fix; not an
SDK-compatibility break.

## `/v1/messages` and `/v1/responses` — static review, NOT live-verified

The two non-chat routes were read against their source
(`src/tilerl/messages.py`, `src/tilerl/responses.py`) but **not** exercised with
live streams on 2026-09-16 (GPU was reserved). Everything below is a
**【静态·待真机 / static, pending live】** source-only conclusion; treat the
live-verified OpenAI notes above as higher confidence. Risks, highest first:

1. **【静态·待真机】 Responses reasoning item may be a stale beta shape.** It is
   emitted as `summary: []` plus `content: [{type: "reasoning_text", text}]`
   (`responses.py` `_output_items`). Current OpenAI GA models the reasoning item
   on `summary: [{type: "summary_text", text}]`; the `content`/`reasoning_text`
   form was an earlier beta. A current SDK that reads only `summary` could see no
   reasoning. Most-likely-real-debt item; verify with the openai-python Responses
   client before changing anything.

2. **【静态·待真机】 No reasoning deltas during the Responses stream.** The
   `content_part.*` events are emitted only for `message` items; a `reasoning`
   item is added with `content: []` and has no incremental event, so the thinking
   text appears only in the final `response.completed` body. Source-visible gap;
   whether the SDK tolerates it is unverified.

3. **【静态·待真机】 Both streaming routes replay a finished body, they do not
   stream tokens as generated.** Source-visible shape: each route's SSE generator
   first awaits the full completion (`await_completion`) and only then emits the
   block/item events; it is not a token-by-token forward of the engine. Stated
   narrowly as a code-reading observation, not a measured behaviour. Consequence
   visible in the source: text/tool payloads are one full-frame delta per block,
   and a mid-run failure is returned as an HTTP 4xx/5xx **before** the SSE
   response starts (there is no in-stream error event). No `ping` / keepalive
   event is emitted on either route, so a long cold prefill sends no bytes for
   tens of seconds.

4. **【静态·待真机】 `reasoning_tokens` is always 0.** Responses `usage.output_tokens_details.reasoning_tokens` is hardcoded 0 even when thinking is
   produced, so reasoning-token accounting under-reports.

5. **【静态·待真机】 Enum/block coverage is the model's actual surface, not the
   full vendor set.** Messages emits only `thinking`/`text`/`tool_use` blocks and
   `stop_reason` ∈ `end_turn`/`max_tokens`/`stop_sequence`/`tool_use` (no
   `pause_turn`, `refusal`, redacted/server/web tool use, multimodal output); the
   thinking block carries `signature: ""` by design (not for replay to real
   Anthropic). Responses emits only `reasoning`/`message`/`function_call` items
   (no web/code/image/local-shell calls). Consistent with a text model; noted so
   a client expecting the extra variants is not surprised.

6. **【静态·待真机】 Messages usage appears on both `message_start` and
   `message_delta`.** The start message spreads the whole completed body (with
   full `usage`) under `content: []`, and the terminal `message_delta` carries
   `usage` again. Whether Claude Code double-counts or takes the terminal value is
   unverified.

Shapes that static inspection finds **correct**: the Anthropic error envelope
`{type:"error",error:{type,message}}` and 400/503 types; tool_use
`input`/`id`/`name` and the `input_json_delta.partial_json` block-delta
placement; Responses `sequence_number` on every event, the
created/in_progress/output_item.*/completed event names, `call_id` on
function_call, and `status:"incomplete"` + `incomplete_details.reason:
max_output_tokens` for a capped turn.

## Live verification still owed

To run in a coordinated idle window after the in-flight server work deploys; file
a fix issue **only if one goes red** (no speculative issues):

1. openai-python **Responses** client: does reasoning parse from the current
   item shape, and is thinking visible during the stream or only at completion?
2. **Claude Code** against `/v1/messages`: is token usage double-counted given
   usage is present at both `message_start` and `message_delta`?
3. **Cold long-prefill SSE** (tens of seconds of silence, no `ping`): do the
   Anthropic and Responses clients tolerate the wait, and does the replay-not-
   stream behaviour cause any client-visible stall or timeout?

