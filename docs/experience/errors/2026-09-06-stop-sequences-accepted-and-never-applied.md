# Stop sequences are a core field of all three APIs, accepted and never applied — cpu, 2026-09-06

> Status: FIXED 2026-09-06 by
> [stop sequences matched on decoded text](../wins/2026-09-06-stop-sequences-matched-on-text-not-token-ids.md).
> **The fix named in this entry was not the one that shipped**: `stop_ids` cannot
> work, because a stop string need not begin at a token boundary, so matching
> happens on decoded text and the single-token/multi-token split disappeared. The
> analysis below is left as written — it is what was believed at the time, and the
> reason it was wrong is the finding.

## Context

Tranche (d) turned every accept-and-ignore field into a 400. Most of that list
is provider features we will never run (hosted `web_search`, `code_interpreter`)
or server-side state we do not keep (`store`, `previous_response_id`). Stop
sequences are different in kind: they are a **core field of both published
APIs** — Anthropic's `stop_sequences`, OpenAI's `stop` — and refusing them is
the honest state today, not the design.

## What is actually true

**`/v1/messages` declared `stop_sequences` and never applied it.** The module
already said so at `messages.py:244`, in a comment on the reply it builds:

> `stop_sequences` is accepted and ignored, so the API's "stop_sequence"
> stop_reason never occurs here.

So a client passing a stop sequence got a 200, no signal, and waited for a
`stop_reason` that could not arrive. That comment is the defect describing
itself and being left in place.

**Chat completions and Responses are worse: `stop` is not a field at all.**
Neither `ChatCompletionRequest` nor `ResponsesRequest` declares it, and pydantic
drops an undeclared field silently — measured:

```
ResponsesRequest.model_validate({..., "previous_response_id": "resp_1"}).model_extra
-> None
```

So the value never reaches the handler and cannot even be rejected. "Accepted
and ignored" understates it; the field is discarded before any code sees it.

**The engine is closer to supporting this than the refusal suggests.**
`SamplingParams.stop_token_ids` exists and **is honoured** — `engine.py:1129`
finishes the request when a sampled token is in that tuple. What is missing is
two things, not the mechanism:

1. Nothing plumbs a *request* field into it. `prompt.py:57` fills
   `stop_token_ids` from the tokenizer's own eos ids and from nothing else, so
   the set is fixed per model.
2. It stops on a **token**, and a stop *sequence* is a string that may span
   several tokens and need not align to a boundary. `end_think_ids` shows the
   multi-token pattern already in the loop (it forces a sequence one token per
   tick), so the shape to copy is there, but a stop sequence also has to
   *truncate* the emitted text back to the match, which `end_think_ids` never
   does.

## Fix, named

Plumb `stop`/`stop_sequences` to a new `SamplingParams.stop_ids:
tuple[tuple[int, ...], ...]`, matched in the same `_step` branch that already
reads `stop_token_ids`, and truncate `req.output` to the match start before
finishing so the sequence itself is not returned. Then `stop_reason`
`stop_sequence` (Anthropic) and `finish_reason` `stop` with the matched string
(OpenAI) become reachable. Single-token stops work with the existing
`stop_token_ids` and are the cheap half; the multi-token case is the reason this
is not a one-liner.

## The record-vs-reply contract, decided 2026-09-06

The open question here was that `messages.py` records completion ids for GRPO,
so cutting the reply's text at a stop match while recording untruncated ids makes
the record and the reply disagree. Settled: **the record keeps what the model
sampled, the reply keeps what the API promises.**

- Generation stops at the token that completes the match.
- `completion_ids` are recorded up to and including that token.
- The returned text is cut at the match *start*.
- The JSONL row carries `stop_reason: "stop_sequence"` and the matched string, so
  a reader knows why the text is shorter than a decode of the ids.

The two therefore disagree by at most the tail of one token, which is inherent to
token-level sampling rather than a defect of this design. Order: single-token
stops through the existing `stop_token_ids`, then multi-token through a rolling
decode of the last k tokens against the longest stop string.

## Not established

- **No recorded request has ever sent one.** Neither the chat page nor Claude
  Code sends stop sequences, so the refusal is believed to break no live client
  — but that is an argument from absence over the requests we happen to have
  recorded, not a measurement of what clients will send.
- **What `k` is** in the rolling decode. It is bounded by the longest stop
  string, but no measurement says what that costs per token, and the match has to
  survive a stop string that does not align to a token boundary.

## Rule

An unimplemented field that is core to the API it belongs to gets a 400 **and**
a line in `OPEN.md`, not a quiet place on the unsupported list. The refusal
stops the silent wrong answer; the line is what stops the refusal from becoming
the permanent design by default.
