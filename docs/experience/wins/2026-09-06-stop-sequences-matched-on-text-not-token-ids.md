# Stop sequences, matched on decoded text rather than token ids — cpu, 2026-09-06

> Status: Shipped. Verified live on the V100 (27B NVFP4, sm70) 2026-09-06 on merge sha
> `753da30` — **by three manual probes, not by the e2e arms**: all three
> `api_e2e.py` stop arms SKIPPED there ("reply too short to cut inside"), so the
> automated coverage of this feature against real weights is still zero. See
> [the live run](#the-live-run-three-manual-probes-because-the-arms-skipped).

## Context

Tranche (e), closing
[stop sequences accepted and never applied](../errors/2026-09-06-stop-sequences-accepted-and-never-applied.md).
The three API surfaces declared `stop` / `stop_sequences` only so tranche (d)
could refuse them with a 400. This makes them work.

## The design changed from the one the error entry named

That entry's fix was `SamplingParams.stop_ids: tuple[tuple[int, ...], ...]` —
stop strings pre-encoded to id tuples, matched against the tail of `req.output`
the way `end_think_ids` already is. Ordered as single-token first (through the
existing `stop_token_ids`), then multi-token.

**Both halves of that plan are gone, and the reason is one property of BPE: a
stop string need not begin at a token boundary.** `"is"` inside `" is 4."` may be
carried by a token spelling `" is"`, so comparing encoded ids against a tail of
sampled ids misses the match — the ids differ while the text agrees. Encoding
per stop string cannot fix it, because which token carries the first character
depends on what precedes it, which is not known until generation.

So matching happens on **decoded text**, and single-token stops stop being a
separate case: one path, one implementation, and the "cheap half first" ordering
had nothing left to order. `stop_texts: tuple[str, ...]` on `SamplingParams`,
matched in `_commit` right after the token is appended.

The cost of that choice is one real concession: the engine's docstring said
*"The engine is tokenizer-free"*, and now says *"unless a caller asks for text
stop sequences"*. `Engine(decode=...)` takes the tokenizer's `decode` and
nothing else; `submit` **refuses** `stop_texts` when it is absent, rather than
accepting a stop that can never fire.

## The rolling window, and what k is

`_stop_hit` decodes only the last `k` tokens, `k` = the longest stop in
**characters**. A token carries at least one character, so k tokens always cover
a k-character match; there is nothing to prove per tokenizer.

The open question in the error entry was what k costs. Measured directly, tiny
model, CPU, median of 200 × 100 calls:

| k (chars) | per token |
|---:|---:|
| 3 | 0.55 µs |
| 16 | 0.99 µs |
| 64 | 2.64 µs |

Against a measured 1345 µs decode tick that is **0.04% to 0.20%**. The
end-to-end paired comparison came out at −13 µs/token (stop-on faster), which is
noise around zero and is reported as a bound, not a speedup: the isolated number
above is the claim, the end-to-end one only says the cost does not show up.

## The contract, as settled

- Generation stops at the token that **completes** the match.
- `completion_ids` keep that token — the record is what the model sampled.
- The returned text is cut at the match **start**, so no client sees the
  sequence: OpenAI and Anthropic both exclude it.
- `/v1/messages` reports `stop_reason: "stop_sequence"` and the matched string in
  `stop_sequence`; its JSONL row carries both.
- `/v1/responses` stays `status: "completed"` — a stop is a complete response, so
  `incomplete_details.reason = "max_output_tokens"` would tell a client to ask
  for more when it already got what it asked for.

## A stop matched from token 0 kills every thinking-on request

Caught in review, reproduced on a real engine before fixing: with thinking on, the
reasoning block is part of `output`, so a stop like `"\n\n"` fires on the
**reasoning's first paragraph break**. Measured — stop set to the reasoning's first
byte, the request **ended at token 1**: a truncated thought, no answer, and
`stop_reason: "stop_sequence"` to explain it. That is every thinking-on
`/v1/messages` request carrying a paragraph stop, which is the common case for a
delimiter-based client.

The engine already tracked `thought_closed` for `end_think_ids`, so the fix reuses
it: while the prompt opened `<think>` and the closer has not been seen, nothing
matches; after it, `_stop_hit` reads `output[reply_from:]` where `reply_from` is the
index just past the closer. A request with no `end_think_ids` (thinking off, or the
tiny/dev bare turn) matches from token 0 as before.

**The canned double had to learn the same rule, and that is the second instance of
this entry's finding**: the route arm passes with the engine's gate deleted, because
`_ScriptedEngine` now skips past the closer too. Two arms, one per layer, each red
only when its own layer is reverted.

## Streaming needed a second mechanism, and a holdback alone was wrong

The stop arrives one token at a time, so a stream that forwards each delta leaks
a prefix before the match completes. Holding back `len(longest stop) - 1`
characters covers the forming case — but not the completed one. Measured: with
the holdback alone the frame before the last carried `"The answer "`, the leading
space of `" is"`. The loop needs both: cut at the match once it is complete, hold
back otherwise.

## Controls

Each revert was run with `__pycache__` cleared, since a restored file is not a
restored import.

| reverted | goes red |
|---|---|
| `_commit`'s stop check | `test_the_engine_stops_at_a_text_sequence_and_names_it` |
| `cut_at_stop` returning text uncut | 5 arms across all three routes |
| the stream's cut + holdback | `test_chat_stream_never_emits_the_stop_sequence` only |
| the `thought_closed` gate in `_commit` | `test_the_engine_does_not_stop_inside_the_reasoning_block` (real engine) — the route arm stays GREEN |
| the same rule in `_ScriptedEngine` | `test_a_stop_does_not_fire_inside_the_reasoning_block` (route) |

**The first control is the finding.** With the engine's matching disabled, all
six route arms passed — because `_ScriptedEngine` honours `stop_texts` itself, so
the route arms never exercise `_commit`. A canned engine that implements the
behaviour under test makes the route gates agree with themselves. The real-engine
arm in `test_e2e.py` is what actually holds the engine to the contract, and it
forces the stop from the model's own first byte rather than hoping noise contains
a chosen string.

## The live run: three manual probes, because the arms skipped

Run by 27 against the V100 (27B NVFP4, sm70) on merge sha `753da30`.
`scripts/api_e2e.py`: **rc 0, 11 passed — and all three stop arms SKIPPED**,
`"reply too short to cut inside"`. The arms derive the stop from the model's own
reply as `text[1:4]` and skip below 6 characters; the probe's prompts ask for
one-word answers, so the reply is too short by construction. **The arms cannot
fire as written, which means they have never run against real weights.** Own
entry: [three live arms skipped by construction](../errors/2026-09-06-three-live-arms-skipped-by-construction.md);
recorded here as the reason the evidence is manual.

The three probes, each naming what only real weights could establish:

| # | request | result |
|---|---|---|
| 1 | `/v1/messages`, thinking ON, `stop_sequences: ["\n\n"]`, "Explain in three separate paragraphs why the sky is blue", `max_tokens` 600 | `stop_reason stop_sequence`, `stop_sequence "\n\n"`, **thinking block 1218 chars intact** + text 696 chars = the first paragraph only |
| 2 | `/v1/messages`, thinking disabled, `stop_sequences: [" is"]`, "Complete this sentence in one line: The capital of France" | `stop_reason stop_sequence`, `stop_sequence " is"`, text `"The capital of France"` |
| 3 | `/v1/chat/completions` stream, `enable_thinking false`, `stop: ["Paris"]` | `finish_reason stop`, joined deltas `"The capital of France is "`, `"Paris"` in **no** frame |

**Probe 2 is the one this whole design rests on.** `" is"` inside `" is 4."`-shaped
text is the case where the 27B's BPE can merge the leading space into one token, so
a tail-of-ids comparison would miss it. The match fired and the exact requested
string came back — the mid-token case survives on the real tokenizer, which is
precisely what the CPU gate and the byte tokenizer cannot say.

**Probe 1 is the reasoning gate on real weights**, and it is the probe the automated
suite could never have produced: the 27B emitted 1218 characters of reasoning
containing paragraph breaks, and none of them fired the stop. Before the gate, that
request returned a truncated thought and no answer.

**Probe 3 confirms the streaming cut end to end** — the holdback plus the
complete-match cut, on a model that emits the stop mid-reply rather than at the end.

## Not established

- **Automated coverage against real weights is zero.** Everything above is manual.
  The three `api_e2e.py` arms skipped on the live run and are fixed in a follow-up;
  until that runs, a regression in this feature would be caught by the CPU gates
  only, and those run over a double that implements the behaviour itself.
- **No arm covers the reasoning gate automatically, live or canned-with-weights.**
  Probe 1 was hand-driven with a prompt chosen to provoke multi-paragraph reasoning;
  nothing in the suite provokes it.
- **No cap on how many stop sequences a request may carry.** Anthropic's API
  allows four; we accept any number, and `_stop_hit` is linear in that count.
- **`k` is in characters, not tokens.** For a stop string of mostly multi-byte
  characters the window is wider than it needs to be, which costs decode time and
  never correctness.

## Rule

When a canned test double implements the behaviour under test, every gate above
it passes with the real implementation deleted. Before believing a route-level
green, delete the mechanism in the engine and re-run: if nothing goes red, the
double is answering, and the gate belongs one layer down.
