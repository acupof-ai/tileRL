# The served prompt did not match the checkpoint's own chat template, 2026-09-03

> Status: **fixed.** `render_chat`'s default left the assistant turn bare — neither opening
> nor closing `<think>` — and **that state does not exist in the checkpoint's template**. The
> model therefore opened `<think>` itself, so the tag rendered as answer text in the chat page.
> The reasoning-effort instructions the template injects into a system turn were never
> rendered at all. Separately, the stream's `usage` was hard-coded `None`, so the page
> estimated its own tok/s as `chars/4` — **~4× low for Chinese**.

## How it was found

The user opened the chat page and reported four things at once: not streaming, no rate shown,
`<think>` not separated, and "聊天模板也不对". The first two were a stale server (the streaming
fix was committed but the process predated it). The last two were real, and the template one is
the interesting defect.

## The template, rendered rather than read

`render_chat` (`tokenizer.py`) had three states:

```python
tail = {None: "", True: "<think>\n", False: "<think>\n\n</think>\n\n"}[thinking]
```

and `server.py` passed `None` whenever a client sent no `chat_template_kwargs` — which the chat
page never does. So the served prompt ended at `<|im_start|>assistant\n`.

**The checkpoint has no such state.** From `chat_template.jinja`:

- line 46: `{%- if enable_thinking is undefined or enable_thinking is true %}` — undefined means
  **on**.
- lines 165-168: the tail is always either `<think>\n\n</think>\n\n` (explicitly off) or
  `<think>\n`. There is no third branch.

So a bare turn hands the model a position where it has never seen a prompt end, and it does the
locally likely thing: emits `<think>` as its first token. That is exactly what the page showed.

Rather than trust my reading of the jinja, I rendered it with jinja2 on the pod. The default:

```
'<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the
task, validate key assumptions, consider plausible alternatives, and prioritize
correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n
hi<|im_end|>\n<|im_start|>assistant\n<think>\n'
```

**Two things we were not doing**: the trailing `<think>\n`, and a whole system turn.

All four modes, from the same render, now match byte for byte:

| kwargs | system turn | tail |
|---|---|---|
| *(none — the default)* | xhigh instructions | `<think>\n` |
| `reasoning_effort: low` | low instructions | `<think>\n` |
| `reasoning_effort: medium` | **none** (accepted, carries no instructions) | `<think>\n` |
| `enable_thinking: false` | none | `<think>\n\n</think>\n\n` |
| a caller system turn | instructions **prepended inside it**, not a second turn | `<think>\n` |

Unknown effort raises, the way the template raises.

## Keeping the dev path

`None` is still needed: the tiny/dev path uses `ByteTokenizer`, which has no `<think>` token to
open. So the server picks the default by **asking the tokenizer** rather than by configuration —
`len(tokenizer.encode("<think>")) == 1`. Checked on the card: `<think>` is a single id **248068**
in the real vocab (`</think>` is 248069), and 7 raw bytes under `ByteTokenizer`.

## The fabricated rate

`_chat_chunk` set `"usage": None` on every frame, so the page had nothing to divide by and used
`chars/4`. For English that is roughly right; **for Chinese one character is about one token**,
so the meter read about a quarter of the truth on exactly the content the user was typing.

Fixed by emitting a final usage-only chunk carrying the engine's own `len(output_ids)`.
**Opt-in** behind `stream_options.include_usage`, because that chunk has no `choices` and an
unconditional one **broke two existing tests in this file** — a client indexing `choices[0]`
every frame raises `IndexError` on it. The page reads `obj.usage` before touching `choices`.

## Rule

**A template is executable, so execute it — do not read it.** Both halves of the defect were
things the jinja states plainly and I would have gotten wrong from the prose: that an undefined
`enable_thinking` means *on* (not *neutral*), and that thinking-on injects a system turn at all.
Rendering it took one command and produced the expected strings the tests now assert, instead of
my paraphrase of them.

Second: **the `None` state was invented to serve a test double and then shipped to the real
model.** It exists for `ByteTokenizer`, which genuinely needs it — but it became the *default*
for every client that did not opt out, so the dev path's compromise silently became production
behaviour. The gate is now a property of the tokenizer in hand, not a default argument.

Third, on the fake rate: **a number the client computes about the server is a number the server
should have sent.** `chars/4` is not an approximation of the token count, it is a different
quantity that happens to be close in English.

## Gate

197 passed, 4 skipped, ruff clean. Every expected prompt string in
`tests/test_tokenizer.py` came from rendering the checkpoint's `chat_template.jinja` with
jinja2 on the pod, not from reading it. The new `tests/test_server.py` case pins both halves of
the usage contract: absent without opt-in and every frame carrying a choice, present with an
empty `choices` list when opted in.

**Not yet measured on the card**: the corrected prompt changes what the model sees, so the
served token rate and the acceptance rate could both move. That is a separate arm and needs the
server restarted on this commit; the B=1 curve in
[`wins/2026-09-03-single-stream-b1-baseline.md`](../wins/2026-09-03-single-stream-b1-baseline.md) was
measured on the bench path, which does not go through `render_chat`, so it is unaffected.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | pod | — | qwen38-27b template | default `enable_thinking` | **true** (template line 46) |
| 2026-09-03 | (this) | pod | — | qwen38-27b template | tail we emitted vs template's | **`""` vs `"<think>\n"`** |
| 2026-09-03 | (this) | pod | — | qwen38-27b template | system turn we emitted vs template's | **none vs xhigh instructions** |
| 2026-09-03 | (this) | pod | — | qwen38-27b tokenizer | `<think>` token id | **248068** (single id) |
| 2026-09-03 | (this) | pod | — | qwen38-27b tokenizer | `</think>` token id | 248069 |
| 2026-09-03 | (this) | Mac | cpu | tiny | `<think>` under ByteTokenizer | **7 ids** (one per byte) |
| 2026-09-03 | (this) | Mac | cpu | tiny | page's tok/s error on Chinese | **~4× low** (`chars/4`) |
| 2026-09-03 | (this) | Mac | cpu | tiny | tests broken by an unconditional usage chunk | **2** |
