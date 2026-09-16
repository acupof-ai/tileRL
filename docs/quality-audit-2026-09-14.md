# Quality audit 2026-09-14

Ten-dimension adversarial audit at baseline `6094e41b` (some findings already
disposed by later PRs; state column marks those). 54 agents: one finder per
dimension, every finding independently re-derived by a skeptic told to refute.
**35 confirmed, 11 refuted.** Confirmed findings only; refuted claims are not
recorded here. Line numbers were accurate at 6094e41b and must be re-read
before a fix.

Legend: 🔴 correctness in production · 🟠 gate/test that cannot see the defect ·
🔵 API surface a client hits · ⚪ resource/data · ⚫ docs/scripts/debt.

## 🔴 Production correctness

1. **Sparse full-prefix hit without a draft hangs forever** —
   `engine.py` zero-residual adoption (`prefill_from == len(tokens)`) is gated
   on `self._draft is not None`; a sparse engine with no draft (`--sparse-k`
   without `--draft`) takes `prefill_from = matched`, no chunk ever forwards,
   the row spins until the 1800 s completion timeout (504), and the slot, hot
   blocks and shared-prefix refs leak. Reachable by resending any
   block-aligned prompt (retries, repeated first turn, n>1). Fix: the
   re-forward-last-page path must run for the no-draft sparse follower too.
2. **Phantom block demand off-by-one at aligned decode crossings** —
   `engine.py _build_plan` growth formula covers through position
   `seq_len+q-1`, but the last physical write is `seq_len+q-2` (position
   `seq_len-1` is a rewrite). At a saturated pool it demands one block that is
   never written (~1/16 of aligned-ending final ticks, trigger
   `(prompt+max_new)%16 == 1`), evicts live prefix entries and finishes the
   request with RequestFailed one tick before completion. The draft loop uses
   the correct `<= seq_len-1` bound. Dense-only.
3. **Completion timeout/error routes return without `engine.cancel`** —
   504/RequestFailed paths in the non-stream routes release the HTTP response
   but the engine row runs to max_new_tokens, holding slot and blocks.
   Distinct from the client-disconnect path fixed in #598: here the client is
   still attached and the server gives up by timeout.
4. **SSE GeneratorExit cancel has no end-to-end gate** — the handler calls
   `engine.cancel` on GeneratorExit; no test drives a real mid-stream socket
   close, so it is unverified behavior (source-read only).

## 🟠 Gates that cannot see the defect

5. The old non-stream disconnect gate cancelled the ASGI task instead of
   feeding `http.disconnect` (mechanism gap fixed by #598, with new
   delayed/at-start/spin gates) — the residual coverage gaps are #3 and #4
   (timeout path, SSE/ws), keep this class in mind.
6. **sm90 fused-prelude "gate" prints instead of asserting** and skips on
   every CI runner: it can never go red. — **CLOSED 2026-09-16.** #663 made the
   sm90 test build both preludes and assert (by mean error) that the fused
   `attn_prep` is strictly closer to the f64 oracle than the discrete chain, with
   non-vacuity guards; the sm90 arm then ran unskipped green on H20 card 0
   (`/work/tl013`, torch 2.11.0+cu129): discrete/fused mean-error ratio **1.9527**
   over 3579 differing elements. See the wrap-up closure §1.2.
7. **`/health` lock gates are wall-clock pass/fail with no skip** on a live
   round trip — contradicts the flaky-test inventory.
8. **Distributed `*_world*` gates never run their negative-control flags**;
   the controls proving each gate can fail are manual-only.
9. **`test_layering` does not resolve absolute `from tilerl...` imports**
   although its docstring says it does.
10. **`/ws/chat` has no behavioral disconnect gate**;
    `test_the_routes_cancel_when_the_client_hangs_up` greps source only.

## 🔵 API surface

11. **Tool round-trip transcript destroyed**: `assistant.tool_calls` and
    `role:'tool'` results are not rendered into ChatML.
12. **Responses API silently drops typed `input_text` parts**: documented list
    input renders an empty user turn.
13. **`tool_choice: 'none'` accepted but ignored**: tools still rendered,
    model can still emit a tool call.
14. **reasoning effort caps only on the OpenAI chat route**; the messages and
    responses routes map effort into prompt text with no engine token cap.
15. **Streaming never parses tool calls**: raw `<tool_call>` XML lands in
    content and finish_reason `length`, where the non-stream route returns
    structured `tool_calls`.
16. **Chat route accepts provider-hosted tool types (web_search…) the Responses
    route rejects**, rendering a null-name tool definition.
17. **`submit()` admission is unbounded**: `_waiting` grows without limit and
    waits past the client deadline; no backpressure.

## ⚪ Resource / data

18. **`calibration._row` ignores its target and hardcodes `sm90`**: a V100
    re-calibration writes sm70 measurements into the sm90 population.
19. **Corrupt `KvBootStore` entry raises RuntimeError instead of returning
    None** and fails every running request; blocks leak if aux.pt load fails
    after the copy try-block.
20. **`select_pages` imported from `tilerl_kernels.reference` on the sparse
    hot path**, bypassing the Backend op seam (placement decision owed in
    architecture steps 6/11).
21. **KV fp8 quant/dequant lives in the framework with no sm90 parity gate**
    for its kernel twin.
22. **Quest `page_bounds`/`page_bound_scores` are sm70-only registry cells**
    with no target-neutral CPU twins registered or parity-gated, and the
    Backend methods are unused by production.
23. **Framework reaches into backend privates** `_MAX_VERIFY_W` and
    `Backend._dp_pg`.
24. **`_benchrec` loads `scripts/benchrec.py` by filesystem path**: a wheel or
    sdist install (scripts/ omitted) crashes bench/ledger. Already decided:
    bridge lives in ledger.py with a ponytail marker; package benchrec.py
    after the scripts sweep (architecture "C").
25. **Dead `HostKvPages._evict_to_ssd`**, uncalled since #525 and stale vs
    live spill accounting.

## ⚫ Docs / scripts / tooling

26. Docs index entry counts ~2.3x stale (263 / 127 wins / 136 errors claimed
    vs 592 / 275 / 317 present).
27. README Status says 14 open defects; OPEN.md has 9.
28. design-rl-stack.md says `serve --devices` ships; deleted in #399.
29. design-parallel.md "what already exists" table cites line numbers that no
    longer point at the named symbols.
30. serve-v100.md documents the /health event-loop stall as live after #577
    fixed it.
31. `pod_session_selftest.sh` is red on the clean tree and no gate runs any
    `scripts/*_selftest.sh`.
32. `audit_scripts_entrypoints.py` false-DEAD: its scanner misses
    `spec_from_file_location` path loads.
33. `bench_gil_yield_compare.sh` rewrites `src/tilerl/engine.py` in place;
    both arms are identical after #568 and its restore mutates a tracked file.
34. `verify_h20_fp4.py` / `probe_kv_fp8_27b.py` default `--source` to the
    checkpoint deleted 2026-09-10 and ignore `TILERL_QWEN38_SOURCE`.
35. Completion wait poll interval `0.02` duplicated as independent literals
    across the routes (fold into the 8b unification).

## Refuted (11) — recorded so they are not re-raised

Card-ownership regex drift; BLOCK_TOKENS hardcode in generate/train; wait-loop
"four copies" (three, and 8b already owns them); sparse eager promotions leak;
pad-row half-allocation leak; Engine.shutdown cold-tier leak; AnyIO limiter
starves /health; framework replays `_tl_layout`; pod.sh/k8s/Dockerfile glue
dead; and two review-process meta findings.

## Fix order proposed

- **Now (correctness, before any serving deploy beyond 62adb8c2):** 1, 2, 3.
  Each needs a failing behavioral gate first (1 and 2 CPU-engine gates; 3 a
  route-level timeout-cancels gate), then V100 real-curl verification for 1/2
  alongside the #598 probe.
- Next: API surface 11–16 (client-visible correctness), gates 6–10.
- Fold-ins: 20/23/24 ride the architecture steps; 25 and 33 ride the scripts
  cleanup; 26–30 one docs PR; 35 rides step 8b.

## Reproduction pack (2026-09-14 late, re-verified on 3026d507)

API-surface findings 11-13/15-17 re-derived after #598/#608/#614 — all six
CONFIRMED with minimal repros and owner modules. Fix queue for the route owner
(fixmisc) after step 11; finding 17 is engine:

11. tool transcript: ChatMessage carries only role/content; assistant.tool_calls
    and role:'tool' render empty / leak `<|im_start|>tool`. Fix ChatMessage +
    blocks_to_text (tool_calls→render_tool_call, tool→tool_response).
12. Responses `input_text` part: blocks_to_text returns "" → 400 empty prompt.
    Map input_text→text in responses._to_messages or blocks_to_text.
13. `tool_choice:'none'` accepted but tools still rendered. Set tools=None for
    render when named none (chat + responses/messages share _render).
15. Streaming never parses tool calls: raw XML in content, finish_reason
    length, vs structured tool_calls non-stream. Parse at terminal delta;
    check responses SSE too.
16. Chat accepts web_search-style hosted tools (null-name render) while
    responses refuses them; port the shared _hosted_tools refusal.
17. submit() _waiting is unbounded (engine.py); add a limit + typed exception
    mapped to 429/503.

## Wire shape table for findings 11–16 (2026-09-15, read off b916c50b)

The implementation contract for the two fix PRs (A: 11+12 transcript/parts;
B: 13+16+15 choice/hosted/streaming). Keys/types dumped from the emitting code,
not from client docs. Arguments on every OpenAI-shaped route are a JSON STRING
via `json.dumps(args, ensure_ascii=False)` — space after colon
(`'{"command": "ls"}'`), locked by
`test_chat_tools_come_back_as_tool_calls`; the new streaming delta must be
byte-identical to the non-stream message for the same reply (shared fixture).

### (a) non-stream chat — `POST /v1/chat/completions`

Top: `id` `chatcmpl-<rid>`, `object` "chat.completion", `created` int,
`model`, `choices[1]`, `usage`
{`prompt_tokens`,`completion_tokens`,`total_tokens`}, `system_fingerprint`.

`choices[0]` = {`index`:0, `message`, `logprobs`: null |
{`content`:[{`token`:str,`logprob`:float|null}]}, `finish_reason`:str}.

`message` = {`role`:"assistant", `content`: str|null,
`reasoning_content`: str|null (always present), `tool_calls`: array|null}.

- `content` is null only when tool_calls exist and prose is empty.
- tool_calls[i] = {`id`:`call_<rid>_<i>`, `type`:"function",
  `function`:{`name`:str, `arguments`: JSON str}}.
- finish_reason precedence: "tool_calls" → "stop" (stop text matched) →
  "length" (n_out >= max_new) → "stop".

### (b) chat SSE — `stream=true`

Frame: `data: {id, object:"chat.completion.chunk", created, model,
choices:[{index:0, delta, logprobs:null, finish_reason}], usage:null|obj,
system_fingerprint}\n\n`; ends `data: [DONE]\n\n`.

Current order: delta {`role`:"assistant"}; zero+ {`reasoning_content`} /
{`content`} incremental slices; terminal chunk delta {} +
finish_reason "stop"|"length". `include_usage` adds cumulative `usage` per
frame plus a final choices:[] usage-only chunk.

Target under finding 15 (parity with (a), same ids/arguments): after the
prose content deltas, a chunk with delta {`tool_calls`:[{`index`:int,
`id`:`call_<rid>_<i>`, `type`:"function",
`function`:{`name`,`arguments`: same JSON str}}]}; then the empty-delta
chunk with finish_reason "tool_calls". The call XML is stripped from the
content tail — no `<tool_call>` substring may occur in a content delta.
WS `/ws/chat` done frame ({`t`:"done",`finish_reason`,`usage`}) gains an
additive `tool_calls` key in the same PR; `web/README.md` documents that
frame and must be updated in the same commit. The playground ignores
unknown keys; existing node tests are the gate, no UI change.

### (c) responses — `POST /v1/responses`

Body: `id` `resp_<rid>`, `object` "response", `created_at` float, `model`,
`status` "completed"|"incomplete", `incomplete_details`: null|
{`reason`:"max_output_tokens"}, `error`:null, `output` list,
`parallel_tool_calls` bool, `tool_choice` str (echoed or "auto"), `tools`
list, `instructions`, `metadata` {}, `temperature`, `top_p`, `usage`
{`input_tokens`,`output_tokens`,`total_tokens`,
`input_tokens_details`:{`cached_tokens`:0},
`output_tokens_details`:{`reasoning_tokens`:0}}.

Call output item: {`id`:`fc_<rid>_<i>`, `type`:"function_call",
`call_id`:`call_<rid>_<i>`, `name`:str, `arguments`: JSON str,
`status`:"completed"}. Message parts use type "output_text".

SSE (every event carries top-level `sequence_number`):
`response.created` / `.in_progress`; per output index
`response.output_item.added`; for a message:
`response.content_part.added` → `response.output_text.delta` (full text,
`logprobs`:[]) → `.done` → `content_part.done`; for a call:
`response.function_call_arguments.delta` → `.done`; then
`response.output_item.done`; final `response.completed`. Stream replays the
parsed body, so calls are already structured — finding 15 needs a gate here,
no code.

### (d) messages — `POST /v1/messages`

Body: `id` `msg_<rid>`, `type` "message", `role` "assistant", `model`,
`content` blocks, `stop_reason`
"tool_use"|"stop_sequence"|"max_tokens"|"end_turn", `stop_sequence`:
str|null, `usage`
{`input_tokens`,`output_tokens`,`cache_creation_input_tokens`:0,
`cache_read_input_tokens`:0}.

Blocks in order: {`type`:"thinking",`thinking`:str,`signature`:""} if
reasoning; {`type`:"text",`text`:str}; one per call
{`type`:"tool_use",`id`:`toolu_<rid>_<i>`,`name`:str,`input`: OBJECT} —
`input` is the coerced dict, not a JSON string (differs from the OpenAI
routes).

SSE: `message_start` (content:[]) then per block
`content_block_start` (tool_use opening has `input`:{}) →
`content_block_delta` (`input_json_delta.partial_json` = json.dumps(input)
for tool_use; `thinking_delta` / `text_delta` otherwise) →
`content_block_stop`; then `message_delta`
{`delta`:{stop_reason,stop_sequence},`usage`}; `message_stop`.

### Incoming shapes (additive only)

- chat `ChatMessage` today: {`role`:str,
  `content`:str|list[dict]|null}. PR A adds optional
  `tool_calls`:[{`id`?,`type`:"function",
  `function`:{`name`,`arguments`: JSON str}}] and `tool_call_id`:str;
  `role`:"tool" turns carry string content and render with the existing
  `blocks_to_text` `<tool_response>` wrapper byte-for-byte.
- responses input items already supported: message items with block
  `content`, `function_call` {name,arguments}, `function_call_output`
  {output}. PR A aliases part types `input_text`/`output_text` to text.
- messages incoming `tool_use`/`tool_result` blocks already render via
  `blocks_to_text`.

### Reusable test fixtures

`tests/test_server.py` (tiny engine + TestClient): fixtures `client`,
`model_id`; `_ByteTokenizer` (1 token/byte, leading id 1),
`_TextTokenizer` (fixed-pattern prefix decode), `_text_blocks`;
`_ScriptedEngine(tokenizer, replies[])` — submit-order canned replies,
two-stage `peek`, stop_text emulation, logprobs/stats/room_for. Relevant
gates: `test_messages_tool_use_round_trip`,
`test_parallel_tool_calls_become_separate_blocks`,
`test_the_sse_stream_keeps_the_shape_a_reader_has_to_handle`,
`test_sse_frames_survive_a_separator_splitlines_cuts_on`,
`test_messages_stream_is_anthropic_sse`.

`tests/test_api_sdk.py` (official openai + anthropic SDKs over uvicorn):
`_PromptKeyedEngine(_ScriptedEngine)` returns the canned TOOL_CALL reply
keyed on prompt content and records `engine.prompts[]`; one module-scoped
server already serves stream and non-stream, so it is the shared fixture
for stream/non-stream parity. Fixtures `oa`, `an`, `base_url`, `engine`;
consts `TOOL_CALL` (Bash/ls), `PLAIN`, `THINKING_ON`. Gates:
`test_chat_tools_come_back_as_tool_calls` (locks arguments spacing),
`test_chat_stream_reconstructs_the_same_text`,
`test_responses_tool_call_and_replay`,
`test_responses_stream_events_and_order`,
`test_messages_tool_use_round_trip`,
`test_chat_refuses_a_tool_choice_it_cannot_force`,
`test_responses_refuses_a_field_it_cannot_honour`.
