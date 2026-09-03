# The reasoning path was dead: addThinking was called but never defined, 2026-09-03

> Status: **fixed.** The `<think>` splitting I shipped in `69398d4` calls `addThinking(bubble)`,
> and **that function does not exist**. Every response containing `<think>` died mid-stream with
> a `ReferenceError`. The checkpoint's template defaults thinking ON, so that is **every
> response**. Found by the subagent redesigning the page, not by me, and not by any test.

## What happened

The commit before this one fixed the served prompt so the model no longer emits a bare `<think>`
tag, and added page code to route the reasoning into its own element:

```js
if (!think) think = addThinking(bubble);
```

`addThinking` was never written. I had `addMsg` and `addEvent` in front of me, wrote a call to a
third one by analogy, and never ran the page.

**The blast radius is total, not partial.** The template's default is thinking on
(`errors/2026-09-03-served-prompt-did-not-match-the-checkpoint-template.md`), so every single
reply enters that branch as soon as `<think>` appears in the accumulated text — which is at the
first delta. The stream then throws inside the `readSSE` callback and the turn dies with a
blinking cursor.

## Why no test caught it

The whole gate is Python. `tests/test_server.py` asserts on the SSE bytes — delta count, joined
text, no U+FFFD mid-stream, the usage contract — and every one of those passes, because the
**server** is correct. The defect is in a 21 KB JavaScript string that Python only ever checks
for length.

So the CPU suite proves the wire format and proves nothing about the page. That was true before
this bug and is still true: there is no JS execution in CI. The subagent caught it by extracting
the served JS and running it against a stub DOM, which is the check that was missing.

## Rule

**A string is not code until something runs it.** `_CHAT_UI` is 21 KB of JavaScript living
inside a Python `"""` literal, so every tool in the repo treats it as data: ruff lints the
Python around it, pytest imports it and measures its length, and `git diff` shows it changed. A
call to an undefined function is the most basic error a language can report, and nothing in the
pipeline was in a position to report it.

Second, and the reason I did not notice: **I verified this change on the wire and called it
verified.** The commit message says "197 passed, ruff clean" and both are true. The SSE stream
was measured on the card, the deltas counted, the Chinese checked for replacement characters —
all of it upstream of the line that was broken. **A gate that passes tells you what it covers,
not that the change works.**

## Also fixed in the same pass

The subagent found and fixed one of its own: on an HTTP error the `throw` sat outside the `try`,
leaving the cursor stuck on a dead turn.

## Gate

197 passed, 4 skipped, ruff clean. `addThinking` verified present and called (2 sites). The
served JS was executed against a stub DOM by the subagent, with SSE frames fed in 7-byte chunks
to force mid-tag splits — the case that made the think/answer partitioning necessary in the
first place. Server restarted on the fix; a live streamed request returns 6 deltas and the
answer no longer carries a literal `<think>` tag.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 69398d4 | — | — | — | `addThinking` definitions in `_CHAT_UI` | **0** (called at 1 site) |
| 2026-09-03 | 69398d4 | — | — | — | replies hitting the dead branch | **all** (template defaults thinking on) |
| 2026-09-03 | 69398d4 | Mac | cpu | tiny | Python tests that could see it | **0 of 197** |
| 2026-09-03 | (this) | — | — | — | `addThinking` after the fix | **1 definition, 2 calls** |
| 2026-09-03 | (this) | V100 | cuda | qwen38-27b | live stream after the fix | 6 deltas, no literal `<think>` |
