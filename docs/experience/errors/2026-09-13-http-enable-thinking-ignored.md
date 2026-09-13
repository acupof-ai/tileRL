# The OpenAI chat route silently ignored a top-level enable_thinking — 2026-09-13

> Status: fixed on 2026-09-13.

## Context

The V100 dense serve's 50-question MMLU smoke (`serve_mmlu_smoke.py`,
2026-09-13) returned 50/50 HTTP 200 with empty `content`,
`finish_reason=length` at 32 tokens, accuracy 0/50. Every completion token was
in `reasoning_content` — the model spent the whole budget thinking. The client
sent the sglang/OpenAI-shaped `"enable_thinking": false` at the request's
top level. A direct-answer prompt and the `chat_template_kwargs` placement both
worked, so transport and the model were fine.

Evidence (Qwen3.8-27B, V100 sm70, one MMLU item, max_tokens 64):
top-level `enable_thinking: false` → `content=""`, populated
`reasoning_content`, finish `length`; `chat_template_kwargs.enable_thinking:
false` → populated `content`, `reasoning_content=null`.

## Root Cause

Only the websocket path moved a top-level `enable_thinking` into
`chat_template_kwargs` (`_ws_body`). `ChatCompletionRequest` declares
`chat_template_kwargs` but not `enable_thinking` and is configured
`extra="allow"`, so pydantic stashed the top-level field in `model_extra`;
`_submit` read only `chat_template_kwargs`, found nothing, and fell back to the
checkpoint template's default — thinking ON for a tokenizer that has the
`` tag. The field the client sent never reached the one prompt renderer.

The sibling routes did not share the shape: the Anthropic route reads its own
`thinking:` dict, and the Responses route only accepts `chat_template_kwargs`.

## Fix

One shared `_normalize_thinking(body)` moves a top-level `enable_thinking` into
`chat_template_kwargs`; both the websocket path (`_ws_body` now calls it) and
the HTTP route apply it before model validation.

## Rule

`extra="allow"` keeps an undeclared field for the unknown-field warning but
does not route it: an OpenAI-shaped knob a client puts at the top level must be
explicitly normalized into the place the handler reads, on EVERY route, not
just the one the first client used. A smoke that scores `content` only reads
the field the renderer actually honoured — 50 clean 200s with empty answers
is a routing bug, not a model result.
