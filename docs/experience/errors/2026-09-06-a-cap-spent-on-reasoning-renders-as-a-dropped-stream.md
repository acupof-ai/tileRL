# A cap spent entirely on reasoning renders as a dropped stream — server, 2026-09-06

> Status: no server change — the reply shape is the OpenAI reasoning convention and the page
> already renders it as truncated (`protocol.ts:73`, gated at `tests/test_chat_ui.py:526`).
> #195's omitted-cap default removes the other route into this state.

## Context

ckl reported that streams were frequently dropping. Two causes were found on the two sides of
the wire, and this entry is about a third mechanism, sharper than either, found while debugging
a probe of my own that had failed for an unrelated reason.

The thinking model spends its first tokens on reasoning. Reasoning goes to
`reasoning_content`, not `content`. So a cap that is too small does not truncate the answer —
**it never reaches the answer**, and the reply comes back `finish_reason=length`, tokens spent,
`content: ""`. A client renders that as a blank bubble, which is what a dropped stream also
looks like.

## The measurement

Live V100 server (`--max-batch 1 --max-ctx 32768 --depth 1`, pid 2771010), non-streaming so the
final body is exact, three prompts × seven caps:

| prompt | cap 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| "Name one colour." | EMPTY | EMPTY | **Blue.** | Blue. | Blue. | Blue. | Blue. |
| "What is 17 times 23?" | EMPTY | EMPTY | EMPTY | **17 ×** | 17 × 23 = **391** | ✓ | ✓ |
| "Explain in two sentences why the sky is blue." | EMPTY | EMPTY | EMPTY | EMPTY | **When sunlight…** | ✓ | ✓ |

Every EMPTY cell is `finish_reason=length` with the full cap consumed and `content` an empty
string. The reasoning was there in each case — 35, 51, 41 characters at cap 8 — so the model was
working, and the budget ran out inside the thought.

**Smallest cap that produced any content: 32 (short), 64 (medium), 128 (long).** The threshold
is a property of how much the model thinks about that prompt, so no fixed cap is safe for all
prompts.

## Root cause

Reasoning and content share one `max_new_tokens` budget, and reasoning is emitted first. There
is no reservation, so a cap below the reasoning length yields a syntactically valid reply that
carries no answer. The old default of **512** was above all three thresholds here, which is why
this shape did not appear in normal use — but the demo page shipped a `value="512"` box the user
could edit, and any smaller value silently produces blank replies rather than short ones.

## Fix: none on the server, and that is deliberate

**`content: ""` with `reasoning_content` present and `finish_reason: "length"` is the correct
reply.** It is the OpenAI reasoning-model convention, and API format fidelity is a standing
order on this project — a future reader who treats this entry as a bug report and "fixes" the
server would break the shape every reasoning client expects. Recorded here so that does not
happen.

**The page already handles it, and the case is already gated.** Verified rather than assumed:

- `web/src/protocol.ts:73` `outcome(finish, answer, sawReasoning)` returns `"truncated"` for
  exactly this state — empty answer, `finish === "length"`, reasoning seen — and its own comment
  distinguishes it from an empty answer that finished normally, "a model that chose to say
  nothing, which is not the same bug and should not carry the same notice".
- `web/src/main.ts:46` calls it; `web/src/render.ts:187` `settle()` writes the notice and opens
  the reasoning fold only for `"truncated"`.
- `tests/test_chat_ui.py:526` `test_the_page_explains_a_reply_the_budget_cut_off` asserts the
  empty answer, `foldOpen is True`, and a note containing "budget". **The explicit-small-cap
  case needs no new test — this is it**, driven through the real WS frames with `max_tokens` set
  to the reasoning length.

The omitted-cap default (#195) removes the other route to this state: an omitted `max_tokens`
now gets the context remainder, ~32K on this server, rather than a flat 512.

**What remains unguarded is only the server's silence about it.** A caller who asks for 16 over
the API and does not read `reasoning_content` sees an empty answer and a `finish_reason` that
does not say the budget went to the thought. That is the convention's cost, not a defect.

## What this did not explain

Three candidates were on the list. **This entry's mechanism is the only one that reproduced**;
the other two are refuted, and both are recorded so they are not re-opened on suspicion.

**Candidate 2, concurrent queueing at `--max-batch 1`: real, and it does not drop anything.**
A request arriving 0.3 s into a stream had TTFT **3.19 s against the first request's 3.40 s
total** — queued behind it, as the flag implies. Swept 2, 3 and 4 concurrent arrivals against a
120-line stream: **all arms completed, zero errors**, and the long stream's own total was 0.99x
its quiet control. Queueing delays a request; it does not drop one.

**Candidate 3, WS idle without ping frames: refuted twice, config then measurement.**
`cli.py:202` calls `uvicorn.run(app, host, port, log_level)` and passes no ws arguments, so the
defaults apply — and on the pod's uvicorn 0.52.3 those are **`ws_ping_interval=20.0`,
`ws_ping_timeout=20.0`**, i.e. pings are on, not absent. Read from `uvicorn.Config.__init__`'s
signature on the running interpreter rather than from documentation, because the premise was
that pings were missing.

Then measured, because a default is not a measurement: a socket opened with **no traffic at all**
stayed open through **90.0 s of silence — 4.5 ping intervals — with no frame and no close**, and
then served a request on the same connection (`done finish_reason='stop'`, 16 frames at 91.0 s).
An absent-ping drop would have closed it at roughly `ping_timeout`.

A first attempt at this arm **reported itself inconclusive and was not used**: the generation
finished in 31.3 s, only 1.6 ping intervals, because the model stopped at 400 numbers before the
3000-token cap. Mid-generation traffic is also the wrong shape for an idle-socket question — the
frames themselves keep the connection busy.

**One stall is recorded as unreproduced rather than as a finding.** The first concurrency probe
showed a 0.81 s gap in the in-flight stream, 19.8x the quiet control's 40.8 ms max gap. A
follow-up counting stalls against 4 spaced arrivals found **0 above 5x the quiet mean**, with
the quiet arm also at 0 by the same threshold. One gap that does not reproduce is not a
mechanism.

## Rule

**A probe that reports "0 events" needs its own arms verified before the zero means anything.**
The stall probe printed `arrivals 4, stalls 0` — a clean negative — while all four arrival arms
had failed. They failed because `"Name one colour."` at `max_tokens=8` emits **zero content
frames**, so `ttft` stayed `None`, and my completeness check tested `x.get("ttft")` rather than
`x.get("error")`. The arms recorded no error key, so the verdict line read as measured. A zero
from an arm that never ran looks exactly like a zero from an arm that ran and saw nothing.

**A cap on a thinking model bounds the thought, not the answer.** Any budget arithmetic that
treats `max_tokens` as "how long may the answer be" is wrong by the length of the reasoning,
which is prompt-dependent and unbounded from the caller's side.

**A default read from documentation is not the running configuration, and neither is a
measurement.** Candidate 3 needed both: the signature of `uvicorn.Config.__init__` on the pod's
own interpreter to establish that pings are on at all, and a 90 s idle socket to establish that
they work. The first without the second would have rested on a default the code might have
overridden; the second without the first would not have said why the socket survived.
