# The page split on a closer the server had stripped

**Context.** After #151 was deployed to the V100 (eccac47, 09:40), ckl reported the chat
page still showed no HTML body. Both API routes were re-probed and returned the HTML; the
chat page's own request shape (`stream: true`, `enable_thinking: true`) returned 84
content deltas, finish `stop`, and zero occurrences of `</think>` in the stream.

**Root cause.** Two layers did the same job. `_stream` strips the reasoning block from
every delta (`strip_think(..., opened=True)`), so the page receives only the answer. The
page's `splitThink(raw, inside=true)` looked for `</think>` to find where the answer
began; with no closer it returned `[raw, ""]`: the entire HTML reply went into the
reasoning fold and the bubble under it stayed empty. Before #151 the server's regex never
matched (no opener in the model text), so the closer reached the page and the page's
split worked; #151 fixed the API and broke the page in the same commit, and no gate saw
it because `test_the_reasoning_split_handles_a_reply_that_starts_inside_think` fed
`splitThink` a `</think>` the server no longer sends. Same shape as #155's rounds 5 and
6: a fixture written by hand, asserting an input the live path does not produce.

**Fix.** One layer. `split_think(text, opened)` returns `(reasoning, reply)`;
`_stream` emits the reasoning as `reasoning_content` deltas (vLLM's field) and the reply
as `content`, holding back a possible partial closer while the block is open. The page
drops `splitThink` and keys the fold on `reasoning_content`; a reply whose budget ran out
inside the block (512 tokens by default, all reasoning) now reads "(cut off by
max_tokens before the answer)" instead of nothing.

Gates: `test_the_stream_carries_the_reasoning_as_its_own_field` (server bytes: reasoning
before content, no closer in either, cut-off reply is reasoning only with finish
`length`) and `test_the_page_folds_the_reasoning_the_server_sends_and_shows_the_answer`
(the page's `sendChat` run in node against that same stream). Controls: the stream gate
against the old server fails at `'' == 'planning\n'`; the page gate against the old page
fails at `reasoning == '<p>hi</p>'`, `answer == ''`, which is the reported symptom.

**Rule.** When a response is shaped in two places, delete one. A fixture for a consumer
comes from the producer's real output, driven through the code under test, never typed.
