# HTTP API compatibility surface

Three completion routes share one engine and present three vendor shapes:

- `POST /v1/chat/completions` — OpenAI Chat Completions (streaming SSE and JSON).
- `POST /v1/messages` — Anthropic Messages.
- `POST /v1/responses` — OpenAI Responses.
- `WS /ws/chat` — the browser playground's own transport (not an OpenAI surface;
  see `web/README.md`).

This page records where the OpenAI-flavoured routes match the published schema and
the deliberate, named deviations, so a client integration does not discover them by
crashing. Verification is at three confidence levels, marked per section:

- **live** — exercised against the running V100 serve on 2026-09-16 with
  `openai-python` 2.15.0 and raw `curl`;
- **SDK fixture** — real server-shape payloads validated through the official
  SDK's own models locally (no GPU);
- **【静态·待真机 / static, pending live】** — source reading only.

It is descriptive of the implemented surface; the source of truth remains the
routes in `src/tilerl/`.

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

## `/v1/messages` — Anthropic envelope (SDK-fixture verified, 0 parse-level RED)

On 2026-09-16 the route's real payloads (non-stream body and every SSE event,
including a tool call) were run through the official `anthropic` 1.4.0 SDK's own
pydantic models — `Message`, `Usage`, `MessageDeltaUsage`, and the six
`Raw*Event` models — not read against the docs and guessed. Every shape parsed.
This is **SDK-fixture** confidence: it proves the current SDK accepts the
envelope; it is not a live Claude Code run (one owed item, below).

| Surface | Official Anthropic | Server (`src/tilerl/messages.py`) | Verdict |
|---|---|---|---|
| non-stream content blocks | `text` / `tool_use` / `thinking`; `thinking.signature` required | all three present, ordered thinking → text → tool_use; `signature: ""` placeholder | ✅ SDK-parsed; signature is replay-only and the server is the endpoint, not a replay source |
| `stop_reason` enum | `end_turn` / `stop_sequence` / `tool_use` / `max_tokens` | all four emitted; tool_use first, then stop_sequence, else max_tokens/end_turn | ✅ all four SDK-parsed |
| `stop_sequence` | matched sequence or `null` | the sequence on a match, `null` otherwise | ✅ |
| `usage` fields | `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` (names differ entirely from OpenAI's `cache_write_tokens`) | all four present as integer 0 when no caching | ✅ both `Usage` and `MessageDeltaUsage` SDK-parsed |
| SSE sequence | `message_start` → per block (`content_block_start` / `content_block_delta` / `content_block_stop`) → `message_delta` (`stop_reason` + `usage`) → `message_stop` | identical; tool args via `input_json_delta.partial_json`, thinking via `thinking_delta`, text via `text_delta`; `message_start` carries `content: []` | ✅ all six event models SDK-parsed |

Non-breaking strictness notes (recorded, no issue — none fails parsing):

1. **Streaming is one whole-block delta, not per token.** A UI watching the
   route does not get a typewriter effect; the client accumulates the block in
   one event. Marked `ponytail` at the SSE generator; a per-token forward is its
   own change.
2. **Usage is split across the two events, not repeated.** `message_start`
   carries `input_tokens` and the cache fields with `output_tokens: 0`;
   `message_delta` carries only the final `output_tokens` (#694/#695 — before
   that the same full usage sat on both events and a sum-across-events client
   double-counted, measured live at 16→32 tokens).
3. **`thinking.signature` is an empty string.** A block replayed to the real
   Anthropic API would be rejected, but this server terminates the request and
   never replays; the SDK parse does not fail.
4. **【待真机】 Claude Code token accounting** with usage present at both
   `message_start` and `message_delta` has not been observed against a live
   client. Listed under owed live checks below.

## `/v1/responses` — remaining source-only notes

The Responses reasoning-item and usage gaps that this section used to list were
confirmed **RED on the live V100 serve on 2026-09-16** (issue #687) and fixed by
#690: the reasoning item now carries GA
`summary: [{type: "summary_text", text}]` (legacy `content[reasoning_text]`
kept for older SDKs), `cache_write_tokens` is an integer 0, and
`reasoning_tokens` reflects the extracted reasoning span. The regression probe
is `scripts/probe_responses_reasoning_shape.py`. Still source-only:

1. **【静态·待真机】 No reasoning deltas during the Responses stream.** The
   `content_part.*` events are emitted only for `message` items; a `reasoning`
   item is added with `content: []` and has no incremental event, so the thinking
   text appears only in the final `response.completed` body. Source-visible gap;
   whether the SDK tolerates it is unverified.

2. **【静态·待真机】 Both streaming routes replay a finished body, they do not
   stream tokens as generated.** Source-visible shape: each route's SSE generator
   first awaits the full completion (`await_completion`) and only then emits the
   block/item events; it is not a token-by-token forward of the engine. Stated
   narrowly as a code-reading observation, not a measured behaviour. Consequence
   visible in the source: text/tool payloads are one full-frame delta per block,
   and a mid-run failure is returned as an HTTP 4xx/5xx **before** the SSE
   response starts (there is no in-stream error event). No `ping` / keepalive
   event is emitted on either route, so a long cold prefill sends no bytes for
   tens of seconds.

3. **【静态·待真机】 Enum/block coverage is the model's actual surface, not the
   full vendor set.** Messages emits only `thinking`/`text`/`tool_use` blocks and
   `stop_reason` ∈ `end_turn`/`max_tokens`/`stop_sequence`/`tool_use` (no
   `pause_turn`, `refusal`, redacted/server/web tool use, multimodal output;
   those four stop reasons and three blocks are now SDK-fixture verified above).
   Responses emits only `reasoning`/`message`/`function_call` items (no
   web/code/image/local-shell calls). Consistent with a text model; noted so a
   client expecting the extra variants is not surprised.

Shapes that static inspection and the SDK fixtures find **correct**: the
Anthropic error envelope `{type:"error",error:{type,message}}` and 400/503
types; tool_use `input`/`id`/`name` and the `input_json_delta.partial_json`
block-delta placement; Responses `sequence_number` on every event, the
created/in_progress/output_item.*/completed event names, `call_id` on
function_call, and `status:"incomplete"` + `incomplete_details.reason:
max_output_tokens` for a capped turn.

## Live verification still owed

To run in a coordinated idle window after the in-flight server work deploys; file
a fix issue **only if one goes red** (no speculative issues):

1. ~~openai-python **Responses** client reasoning shape~~ — done 2026-09-16:
   live RED (#687), fixed #690; rerun `scripts/probe_responses_reasoning_shape.py`
   against the deployed serve once to confirm GREEN.
2. **Claude Code** against `/v1/messages`: does the client read the split usage
   correctly — `input_tokens` on `message_start` (with `output_tokens: 0`) and
   the final `output_tokens` on `message_delta` (#694/#695)?
3. **Cold long-prefill SSE** (tens of seconds of silence, no `ping`): do the
   Anthropic and Responses clients tolerate the wait, and does the replay-not-
   stream behaviour cause any client-visible stall or timeout?

