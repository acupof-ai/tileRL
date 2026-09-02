---
question: What does an Anthropic Messages shim in front of tileRL's server actually have to implement, and can the tiny model on CPU complete a `claude -p` tool loop?
status: measured
source: live capture of Claude Code 2.1.258 against a stub Messages endpoint, 2026-09-02; tileRL server.py / tokenizer.py / config.py read the same day
---

# Stage 1: the Messages shim, measured against a real Claude Code request

Answers the three things tilerl-19 stage 1 asks for. **Two of the three
predicted gaps are real; the blocking one was not predicted and is a size
problem, not a schema problem.**

## Method

A stub HTTP server on 127.0.0.1:8899 that logs the request body and returns a
minimal valid Messages response. `ANTHROPIC_BASE_URL` pointed at it,
`claude -p "say hi" --output-format stream-json`. Claude Code completed the
turn and printed `"result":"hi"`, so the transport contract is confirmed end to
end before any shim exists.

## What Claude Code actually sends

`POST /v1/messages?beta=true`, and the top-level keys are:

    context_management  max_tokens  messages  metadata  model
    output_config  stream  system  thinking  tools

Not the shape the OpenAI route accepts (`server.py:44-58`):

| field | what arrives | tileRL today |
|---|---|---|
| `system` | **list of 2 blocks**, each `{type, text, cache_control}` | no field; `render_chat` takes flat `(role, text)` pairs (`tokenizer.py:71-76`) |
| `tools` | **28 JSON-Schema tool defs** | nothing |
| `messages[].content` | **list of blocks**, not a string | `ChatMessage.content` accepts `list[dict]` but nothing renders blocks |
| `thinking` | `{"type": "adaptive"}` | `reasoning_effort` maps to a think budget — closest existing analogue |
| `output_config` | `{"effort": "medium"}` | none |
| `stream` | `true` on the first call | SSE exists but emits one whole-completion delta (`server.py:231-233`) |
| `max_tokens` | 32000 | `max_completion_tokens`, fine |

**Predicted gaps confirmed**: structured `tool_use` output and
`stop_reason="tool_use"`. Neither exists — the OpenAI route returns
`finish_reason` from the engine's stop set and has no notion of a tool call.
Also missing and unlisted: `n`, `stop`, `top_logprobs`.

## The blocking finding: size, not schema

One request with Claude Code's default toolset is **117,226 bytes**. Trimming
to four tools (`--disallowed-tools` for the other 24) gives **21,676 bytes**:

    system   7,106 chars
    tools    6,121 chars
    messages 8,016 chars
    ~5,310 tokens at 4 chars/token

`tiny` has **`max_position_embeddings=512`** (`config.py:218`). The floor for a
real Claude Code turn is ~10x the tiny model's entire context, and almost all
of it is the harness preamble, not the task.

So **stage 1's "done when" cannot be met by the stock CLI against `tiny`**.
Three ways out, cheapest first:

1. **Raise tiny's context.** It is a config integer and CPU RAM at 64 hidden /
   2 layers is not the constraint. Cheapest, and it keeps the CLI unmodified —
   which is the point of the exercise, since the harness preamble is part of
   what we want the policy to learn to operate inside.

   **CORRECTED after running it: 8192 is not enough, 65536 is.** The
   `~5,310 tokens` above is a chars/4 estimate and it is wrong by 9x for the
   tokenizer that actually serves a checkpoint-less tiny. `ByteTokenizer` is
   **one token per byte** (`tokenizer.py:19-24`), so the 21,676-byte trimmed
   request is 21,676 tokens and the untrimmed one is 79,451. The engine
   returned `400 request (79451 tokens) exceeds max_total_tokens (8192)` —
   which is the check working, and the estimate that never should have been
   made in characters. A real BPE would be ~4x denser; the 27B is unaffected
   (262,144 positions).

   Second thing the run exposed: `max_total_tokens` was hardcoded to 8192 at
   `cli.py:72`, independent of the model's context, so raising the config alone
   would still have refused. The budget and the block pool now follow
   `max_position_embeddings`.
2. **Shrink the request**: `--disallowed-tools` plus a minimal system prompt.
   Takes 79,451 tokens to 21,676 — still far over 512, and it changes the
   environment the policy sees, since the trimmed harness is not the one Claude
   Code actually runs.
3. **Serve the 27B instead of tiny.** Correct context, needs the card, and the
   card is held by the pretrain.

**Recommendation: (1) plus (2).** `tiny-agent` at **65536** positions, with the
four-tool trim for the first loop so the pass is about the shim rather than
about generation length. Then re-measure with the full 28 tools (79,451 tokens
byte-level), because that is the real environment.

## Where the shim goes

**tileRL, `src/tilerl/messages.py`, mounted by the same `create_app`.** Not a
separate service and not in aupai:

- it needs the tokenizer and `render_chat` to turn tool blocks into ChatML, and
  those live here (`tokenizer.py:71`);
- the per-request record fb requires (below) needs the engine's `logprobs`
  (`engine.py:534`) and the exact prompt ids — both are in-process here and
  would be a second serialization hop anywhere else;
- aupai is the training-data consumer, not the serving path.

## The per-request record

fb's addition, and it is right for a reason worth stating: BPE is not
concatenation-invariant, so a token sequence rebuilt from transcript *text* is
not guaranteed to equal the sequence that was sampled, and GRPO trained on a
mismatched sequence is a silently wrong gradient. The shim writes one JSONL row
per request:

    {request_id, prompt_ids, completion_ids, logprobs, stop_reason}

and returns `request_id` in a response header so the transcript can be joined
by id rather than by text. Everything needed is already produced —
`Engine.logprobs` pops per-request scores (`engine.py:534-543`), `_submit`
already has the prompt ids — so this is wiring, not new capability.

One caveat to record now: `Engine.logprobs` **pops**. If the shim reads it and
something else expects it later, the second reader gets nothing. The shim must
be the only reader per request.

## State

Built and gated: `src/tilerl/messages.py` mounted on the same app, tests
asserting the wire shape and the record together (header id names the recorded
row, one logprob per generated token, stop_reason agrees, SSE carries the event
names the CLI parses). `Engine.logprobs` now raises on a second read instead of
returning None.

Verified live by curl against a served `tiny-agent`: a real Messages response,
and a record row with 52 prompt ids, 8 completion ids, 8 logprobs — one score
per token, which is the property the RL loop consumes.

The end-to-end `claude -p` pass is **running, not yet passed**. CPU prefill of a
21,676-token prompt through a freshly compiled kernel set is minutes, not
seconds. Reporting it as in-flight rather than as a pass.
