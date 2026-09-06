# The sandbox gate's control was swallowed by the think block — 2026-09-06

> Status: Fixed. Both arms of `test_sandbox_confines_writes_to_the_rollout_dir` now parse the
> tool call; before this, neither did.

## Context

tilerl-25 reported that `test_sandbox_confines_writes_to_the_rollout_dir` fails on clean
`a08b1c2` in a fresh worktree: the negative control does not fire, so the sandboxed half
proves nothing. They flagged it rather than claiming it — "right now the suite has a gate that
cannot fail" — and asked for an owner.

The test is built correctly. It runs the same scripted escape twice, sandboxed and not, and
asserts the file is absent then present, precisely so that neither assertion stands alone.

## Root Cause

**The reply was routed to reasoning, so the tool call never reached the parser — in both
arms.**

Claude Code sends `thinking: {"type": "adaptive"}`, so `_thinking()` returns True and the
27B template opens a `<think>` block **in the prompt**. `split_think(text, opened=True)`
(`prompt.py:131`) then prepends `<think>` to a reply that does not start with one, and with
no `</think>` anywhere the entire reply is the reasoning block and the text is empty:

```
opened=True   -> reasoning carries <tool_call>, reply is ''
opened=False  -> reply carries <tool_call>
"</think>"+x  -> reply carries <tool_call>
```

`messages.py:217` is `_parse_tool_calls(text, req.tools)`, and `text` was `''`.

**What made this hard to see is that everything reported success.** The unsandboxed rollout
returned `is_error: false`, `returncode: 0`, `subtype: "success"`, `stop_reason: "end_turn"`,
`num_turns: 2`, `permission_denials: []`, `result: ""`. Two records were written, so the
engine served both turns. Nothing anywhere says "a tool call was discarded".

## Two wrong answers first, and how each died

**`ANTHROPIC_MODEL` leaking into the child.** `rollout.py:176` builds the subprocess env as
`{**os.environ, ...}`, so an agent session's own model id reaches the spawned CLI, which
rejected it: `[claude-code:unrecognized_model] {"model":"model_hub/es1..."}` in stderr. That
is a real defect and it looked like the cause. **Killed by its own control:** I unset that
variable and the four sibling model vars, held everything else constant, and
`unrecognized_model` disappeared while the file still was not written — same `turns=2`, same
`subtype=success`. A plausible error message in stderr is not the failure's cause just
because it is the only error present.

**"The CLI never dispatched the call."** True but not a cause — it is the symptom one level
up. Instrumenting the seam 25 pointed at settled it: `_parse_tool_calls` **was** called twice,
with 29 tools present, and both times `has_tool_call_tag: false` and `text_head: ""`. So the
parser and the tool list were fine and the text was already gone. Then logging every
`submit` showed both prompts ending in `<|im_start|>assistant\n<think>\n`, which named the
mechanism.

## When it broke, and how long it was uncovered

Dated by execution rather than by commit message: I read `prompt.py` and `messages.py` out of
git at three revisions and ran the live call site's own argument through each.

| revision | `strip_think` has `opened=` | call survives to the parser |
|---|---|---|
| `0dfe018` (before #151) | no | **yes** |
| `eccac47` (#151) | yes | **no** |
| `HEAD` | yes | **no** |

So #151 is the commit, confirmed by running it and not by reading its subject — the test
itself predates it by four days (`a6a88b6`, 2026-09-02), and it worked until `opened=` reached
the live call site. #151's own change was correct and needed; what it did not do was carry the
fixture along with the invariant it introduced.

Uncovered from **2026-09-06T09:37:32+08:00** (`eccac47`) to **T16:29:18+08:00** (the fix),
computed from the two commit timestamps: **6.9 hours**.

## Three of us restated the failure without opening it

Recorded because the pattern is the finding, not the incident:

- tilerl-27 called it "environmental on this Mac" in #159's PR body and in two messages
- tilerl-48 called it pre-existing on clean main
- tilerl-25 said its negative control does not fire here

All three are true statements and none of them is a cause. "The gate fails" restated three
times reads as three independent confirmations, so each restatement made the next reader less
likely to open it — the same accumulation as
[a number with no instrument](2026-09-06-a-number-with-no-instrument.md), where five citations
of one absence looked like corroboration. 25's framing was the one that broke the chain,
because it named which half was unproven ("the sandboxed assertion proves nothing") rather
than reporting a red test.

## Fix

The fixture's scripted reply opens with a closer:

```python
escape = "</think>" + render_tool_call("Bash", {...})
```

That is what a model under this template actually emits — the prompt supplies the opener, the
model supplies the closer — so the fixture now matches the shape the production path already
handles rather than a shape only a test produces.

**Both arms verified to parse the call**, which is the property that was missing:

| arm | tool calls parsed | file written |
|---|---:|---|
| sandboxed | **1** | False |
| unsandboxed | **1** | True |

Before the fix both arms parsed **0**. The sandboxed assertion was passing because no attempt
was made, which is exactly what 25's control was designed to catch, and did.

## Still open, filed separately

`rollout.py:176`'s `{**os.environ, ...}` means every rollout inherits the spawning machine's
`ANTHROPIC_MODEL`, so a child's behaviour depends on which session started it. Not this
failure's cause, and a reproducibility defect regardless.

## Rule

**A test whose fixture bypasses a production transform is green on a path nobody ships.**
The escape XML was valid; what was invalid was emitting it without the closer the template
guarantees. When a fixture hand-builds model output, it has to satisfy the same invariants
the real decoder does, or the assertions measure the fixture.

Second, on the two wrong answers: **an error in stderr is a candidate, not a cause.** The
`unrecognized_model` line was real, alarming and irrelevant, and it survived exactly as long
as it took to vary it alone. **Vary the suspect and nothing else, before writing it down.**

Third, on the three restatements: **a failing test described the same way by N people is one
observation, not N.** "It fails here", "it's environmental", "it's pre-existing" are all
compatible with every cause, so they accumulate confidence without adding evidence. The
report that broke the chain named the *unproven half* rather than the symptom. When repeating
someone else's finding, either open it or say plainly that you have not.

## Results

No runtime change; a test fixture and this entry. The gate it repairs was green-but-vacuous,
so the honest summary is that this repository had no working sandbox-confinement gate until
now — the assertion existed, the coverage did not.
