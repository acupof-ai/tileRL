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

## And then the repaired gate got the same treatment

Both arms parsing the call shows an attempt is made; it does not show the **sandbox** is what
stops it. So the thing under test was reverted one key at a time, each arm required to go red:

| revert | gate | assertion that fired |
|---|---|---|
| none (as shipped) | pass | — |
| `allowUnsandboxedCommands` → `True` | **pass** | none |
| `failIfUnavailable` → `False` | **pass** | none |
| `enabled` → `False` | **fail** | `sandboxed rollout wrote outside its directory` |
| drop `--settings` entirely | **fail** | same |

The confinement is `enabled` plus the payload reaching the CLI. The two keys that left it green
govern *refusal to run where no sandbox exists* — a case this gate never enters — so neither is
a valid control for this assertion, and **the first one I tried was one of them.** Had I stopped
at its green I would have reported a second vacuity that does not exist.

**The docstring was part of the defect.** It called `failIfUnavailable` "the important key",
which is what aimed that first control at the wrong one; tilerl-25 named the sharper half — of
the two, `failIfUnavailable` is the more dangerous, because its name *sounds* like the write
path. A comment that misnames what binds does not merely fail to help, it aims the next
person's control. Rewritten to two facts, one gate named per key, with this history left here
rather than in the source. `failIfUnavailable` also gained the control it never had
(`test_a_host_without_a_sandbox_refuses_rather_than_running_bare`, red on `DID NOT RAISE` when
the refusal is removed).

**A control that leaves a gate green has two readings** — the gate is vacuous, or the key you
reverted is not the one that binds — and only sweeping the candidates separates them. Stopping
at the first green picks whichever reading you already expected.

Both red arms fired on the sandboxed assertion, not on the negative control below it, which is
what makes them controls: a revert that broke the unsandboxed half instead would produce the
same red count and prove nothing.

## Still open, filed separately — now closed here

`rollout.py:176`'s `{**os.environ, ...}` meant every rollout inherited the spawning machine's
environment. Not this failure's cause, and a reproducibility defect regardless, so it is fixed
in the same branch. **Enumerated rather than listed from memory**: this machine exports **20**
matching variables, not the five model vars the original note named —

```
ANTHROPIC_AUTH_TOKEN  ANTHROPIC_BASE_URL  ANTHROPIC_MODEL
ANTHROPIC_DEFAULT_{HAIKU,OPUS,SONNET}_MODEL                          (6)
CLAUDECODE  CLAUDE_PID  CLAUDE_EFFORT  CLAUDE_PLUGIN_DATA
CLAUDE_CODE_{ATTRIBUTION_HEADER,CHILD_SESSION,DISABLE_TERMINAL_TITLE,
  ENTRYPOINT,EXECPATH,MAX_CONTEXT_TOKENS,MESSAGING_SOCKET,
  MESSAGING_TOKEN,SESSION_ID,SUBAGENT_MODEL}                        (14)
```

Two of those are worse than a wrong model id: `CLAUDE_CODE_MESSAGING_SOCKET` and
`CLAUDE_CODE_MESSAGING_TOKEN` are this session's cross-session channel, so a sandboxed rollout
child was being handed the address and credential to message the agent session that spawned it.
Nothing observed it doing so; the point is that an allow-list of five model names would have
left both in place.

The fix strips by prefix (`ANTHROPIC_`, `CLAUDE`) and then sets the four the rollout needs.
Its gate, `test_rollout_env_carries_no_var_from_the_spawning_session`, asserts on the env dict
handed to `subprocess.run` — the CLI's own `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` acts one level
further in and cannot be observed from here — and checks an unrelated `TILERL_KEEP_ME` survives,
so a scrub that took the whole environment would also be red. Its negative control restores
`{**os.environ, ...}` and fails on the leaked-var assertion by name:

```
AssertionError: spawning session's vars reached the child:
  ['ANTHROPIC_MODEL', 'ANTHROPIC_DEFAULT_OPUS_MODEL', 'CLAUDECODE', 'CLAUDE_CODE_MESSAGING_SOCKET']
```

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

Fourth, from the env scrub: **an allow-list of the names you remember is not a scrub.** The
original note said "ANTHROPIC_MODEL and the four sibling model vars" because those were the
five I had seen in an error message; `env | grep` found 20, two of them a live channel back
into the parent session. Enumerate what is actually there before writing the set down.

## Results

No runtime change; a test fixture and this entry. The gate it repairs was green-but-vacuous,
so the honest summary is that this repository had no working sandbox-confinement gate until
now — the assertion existed, the coverage did not.
