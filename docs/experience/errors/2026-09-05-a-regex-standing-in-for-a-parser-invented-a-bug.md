# A regex standing in for a parser invented a bug, and I corrected the person who was right — 2026-09-05

**Date:** 2026-09-05
**Task:** PR 67 review (markdown renderer XSS)
**What I claimed:** the reviewer's fix was half a fix, because escaping only `"` left
`[x](https://a'onmouseover='b)` parsing as a live `onmouseover` attribute.
**What is true:** that variant was never a breakout. My check invented it.

## Context

A reviewer found a real attribute-breakout XSS: `mdEscape` escaped `& < >` and no
quotes, while two rules interpolate its result into an HTML attribute (`href="..."`,
`data-lang="..."`). A `"` closed the attribute and the rest of the token became markup.
Reproduced by execution; the fix is escaping quotes at the source.

They suggested a gate asserting the output "contains no `onmouseover`". That is a
substring search, and it is genuinely wrong here: after the fix the output contains the
inert literal `&quot;onmouseover=&quot;`, which matches. So I wrote a stronger check —
parse attribute *names* out of tag interiors — and with it reported a second live
variant using `'`.

## Root cause

My "parser" was a regex:

```js
for (const tag of html.match(/<[a-z][^>]*>/g) || [])
  for (const m of tag.matchAll(/[\s"']([a-zA-Z-]+)\s*=/g)) attrs.add(m[1]);
```

The character class `[\s"']` treats `'` as an attribute separator. Every attribute this
template writes is **double-quoted**, so a `'` inside the value is an ordinary
character — but the regex matched inside the value and reported an attribute that is
not there. Measured on the exact string, no renderer involved:

| checker | attributes found |
|---|---|
| my regex | `href, onmouseover, target, rel` |
| `html.parser` | `href, target, rel` |

So the mutation harness reported **CAUGHT** for dropping the `'` escape — a kill for a
mutation that reintroduces no vulnerability.

## Fix

The gate feeds each rendered output to `html.parser` and asserts on the attribute names
it returns. The double-quote breakout is still caught; the invented one is gone.

The `'` escape stays as defence in depth — it costs nothing and makes the renderer safe
against the next single-quoted attribute someone writes — but the docstring now says
that, rather than implying it is load-bearing.

The mutation harness now asserts an **expected verdict per case** instead of printing
whatever comes out:

| mutation | verdict | expected |
|---|---|---|
| drop `"` escape | CAUGHT | CAUGHT |
| drop `<` escape | CAUGHT | CAUGHT |
| drop both quotes | CAUGHT | CAUGHT |
| drop `'` escape | passed | passed |

## Rule

**A regex is not a parser, and the moment a check's answer disagrees with a real parser,
the check is the defect.** Where a language has a grammar — HTML, JSON, the shell — use
its parser in the gate. `html.parser` is in the standard library and needed no
dependency.

**Assert the expected verdict in a mutation harness, not just the observed one.** A
wrong kill is worse than a missing one: a missing kill leaves a gap, but a wrong kill
marks inert code as load-bearing, and whoever reads it in six months will not touch a
line that protects nothing.

**Three checks failed the same way in one afternoon, in three directions.** The
reviewer's substring assertion could not tell an attribute from text. My regex invented
an attribute. My attack set was all quote payloads, so dropping the `<` escape read
MISSED until a raw `<img>` went in. Each check only covered the failure its author had
already imagined — which is [scope-the-check-to-the-object] restated: the shape of the
check has to come from the object, not from the hypothesis.

And the specific trap in correcting a reviewer: I had **more** evidence than they did
(I had executed against the live server, they had read the template) and was still
wrong, because the extra evidence went through a broken instrument. More evidence
through a worse instrument is not a stronger claim.
