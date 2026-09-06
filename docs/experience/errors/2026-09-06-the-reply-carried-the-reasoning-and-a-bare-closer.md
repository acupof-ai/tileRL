# The reply carried the reasoning and a bare `</think>`

**Context.** ckl asked why the V100 endpoint "does not output the HTML body". Both routes
were probed at 73bef1d with "write a minimal HTML page":

```
chat:     finish=stop  content='The user wants a minimal HTML page ...\n</think>\n\n```html\n<!DOCTYPE html>...'
messages: stop=end_turn content='User wants a minimal HTML page ...\n</think>\n\n```html\n<!DOCTYPE html>...'
```

The HTML was there. It came after the model's reasoning and a closer with no opener, which
is what a client that renders or parses the reply shows as "no body".

**Root cause.** `strip_think` matched `<think>.*?</think>`. The 27B template opens the
block in the PROMPT (`prompt.py:23`, `thinking=True` appends `<think>\n`), so the model's
own text never contains `<think>`, only `</think>`; the regex found nothing to strip. The
self-check at `messages.py:326` asserted the case with both tags, the one the live path
never produces: it proved what it exercised. `7df722e` "strips reasoning from the reply"
had the same hole from the day it landed.

**Fix.** `strip_think(text, opened=...)`: when the prompt opened the block, the closer
alone delimits the reasoning; the four live call sites (both routes, streaming and not)
pass whether the prompt opened it. Cut-off reasoning with no closer still strips to empty.
Gate `test_a_reply_that_carries_only_the_think_closer_is_the_answer`: opened arm returns
only the answer, thinking-off arm and the byte-tokenizer bare turn pass through untouched;
control (old regex) fails at `'planning\n</think>\n\n<p>hi</p>' == '<p>hi</p>'`. The
messages-route fixtures now start with the closer, as the model's text does.

**Reach (25's review).** Not cosmetic: with the reasoning left in, the Messages route's tool-call
parse saw prose first, so `test_messages_tool_use_round_trip` and
`test_parallel_tool_calls_become_separate_blocks` fail at `'text' == 'tool_use'` once their
fixtures carry the closer the model emits; they were green only because the fixtures never had
the prompt open the block. Streaming (`server.py:286`, `:312`) was broken the same way,
measured through `_StepEngine`: delta `planning\n</think>\n\n<p>hi</p>` before, `<p>hi</p>`
after; `_ScriptedEngine` has no `peek`, so the new test does not reach that path.

**Rule.** Same shape as [an interrupted run reported pass](../wins/2026-09-06-an-interrupted-run-reported-pass.md):
a green assertion about an input the system does not produce. There the test encoded the wrong
answer; here the wrong input.  A self-check for a stripper is written from the string the live path produces,
not from the pair the tag was named after; when the prompt supplies half of a delimiter,
the test input has only the other half.
