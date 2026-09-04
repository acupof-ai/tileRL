# The page's JS now has a gate, and `node --check` is not it, 2026-09-03

> Status: **shipped.** `addThinking` shipped called-but-never-defined
> ([`errors/2026-09-03-addthinking-was-never-defined.md`](../errors/2026-09-03-addthinking-was-never-defined.md))
> and 197 Python tests could not see it. The obvious fix — run `node --check` on the extracted
> script — **does not work**, and I verified that before writing it. What ships instead is a
> syntactic reference check that needs no JS runtime, so CI always executes it. **Negative
> control against the real broken commit reports exactly `['addThinking']`.**

## `node --check` cannot see this class of bug

The first idea was to extract the `<script>` body and hand it to node. It parses:

```
$ node --check /tmp/ui.js
SYNTAX_OK
```

But an undefined call is a **runtime** error, not a parse error. Measured on a two-line file:

```js
function a(){ return undefinedFn(1); }
```
```
$ node --check /tmp/neg.js
SYNTAX_OK -- so --check CANNOT see an undefined call
```

So a `node --check` step would have gone green on the exact commit that was broken. It also adds
a runtime dependency to a gate that must run everywhere. Dropped before it was written.

## What ships

`tests/test_chat_ui.py` — pure Python, no JS runtime:

1. Extract the inline script.
2. **Blank out string literals, template literals and comments** (replaced with spaces, so
   nothing new becomes adjacent).
3. Collect identifiers called **without a leading dot** — a bare call, not a method.
4. Subtract what the script defines: `function` names, `const`/`let`/`var` bindings, arrow
   parameters, and **declared function parameters** (`readSSE(resp, onFrame)` calls `onFrame`).
5. Subtract the browser globals.
6. Fail on anything left.

## Two of my own errors, caught by running it

**Step 4 was incomplete.** The first version flagged `onFrame`, which is a parameter of
`readSSE`. A gate that reports a healthy page is a gate that gets deleted, so this had to be
right before the negative control meant anything.

**Step 2 did not exist**, and the negative control is what exposed it. Against `HEAD~1` the gate
reported:

```
HEAD~1 unresolved: ['addThinking', 'files', 'tilerl']
```

`files` and `tilerl` are not code. They come from UI copy:

```
"read/write files  (Enter to send)"
"Message tilerl  (Enter to send, Shift+Enter for newline)"
```

An English phrase with a parenthetical reads as a call. **Any future placeholder or button
label of the form `word (note)` would have tripped it** — a gate that cries wolf on prose is
worse than no gate. With literals stripped, `HEAD~1` reports `['addThinking']` and nothing else.

## The negative control is a script, not a unit test

`tests/test_chat_ui.py` has a unit-level negative control — the regex must flag a synthetic
`addThinking(bubble)` and must not flag `x.addThinking(1)`. That only proves the pattern
matches a string I wrote.

`scripts/probe_ui_gate_negative_control.py` runs the shipped gate against
`git show HEAD~1:src/tilerl/server.py`, the actual version that shipped broken, and asserts
`addThinking` is in the result. **That is the check that proves the gate is not decoration.**

## What this still does not cover

It is syntactic, so it catches an undefined **call** and nothing else. A typo'd property
(`obj.usge`), a wrong argument order, a CSS class that does not exist, a DOM id that never
matches — all still invisible. The subagent's approach (extract the JS, run it against a stub
DOM, feed SSE frames in 7-byte chunks) covers far more and is the right next step if the page
keeps growing; it needs node, so it would be a skip-if-absent test rather than a blocking one.

`# ponytail: undefined calls only, stub-DOM execution when the page grows`

## Rule

**Reach for the check that runs everywhere over the check that catches more.** `node --check`
felt like the professional answer and is strictly weaker than 40 lines of regex for this
defect — it cannot see undefined calls at all. Testing the tool against the bug before adopting
it took one command.

Second: **a negative control on synthetic input proves the pattern, not the gate.** Both of my
implementation errors survived the unit-level control and died against the real broken commit.
The distinction is worth keeping: the synthetic case pins intent, the historical case pins that
the gate would actually have fired.

## Gate

199 passed, 4 skipped, ruff clean. Negative control against `HEAD~1` reports exactly
`['addThinking']`. `node --check`'s inability to see an undefined call was measured, not assumed.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | Mac | — | — | `node --check` on an undefined call | **SYNTAX_OK (blind)** |
| 2026-09-03 | (this) | Mac | — | — | gate against `HEAD~1`, no literal stripping | `addThinking`, **`files`, `tilerl`** |
| 2026-09-03 | (this) | Mac | — | — | **gate against `HEAD~1`, shipped version** | **`['addThinking']` exactly** |
| 2026-09-03 | (this) | Mac | — | — | gate against `HEAD` | clean |
| 2026-09-03 | (this) | Mac | — | — | false positives from UI copy | **2, now 0** |
| 2026-09-03 | (this) | Mac | — | — | JS runtime required | **none** |
