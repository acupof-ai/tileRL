---
question: What does it take to run one Claude Code episode against tileRL's own server, and does the sandbox that isolates it actually isolate anything?
status: measured
source: measured 2026-09-02 against Claude Code 2.1.258 on this Mac; pod probes via ~/bin/pod the same day
---

# Stage 2: the launcher, and the sandbox that had to be proven

Stage 2 asks for one command that starts the server, points Claude Code at it,
and collects the trajectory — the same command on the Mac and on the pod, with
no Docker. It is `src/tilerl/rollout.py`, gated by `tests/test_rollout.py`.

## The episode tag is a header, and that is a measurement

GRPO's advantage is per **episode**, and one episode is many turns, so every
record row needs to name the episode it came from. Agent Lightning does this
with a per-rollout id; the open question was where to carry it.

Measured, against a stub that dumped its headers: `ANTHROPIC_CUSTOM_HEADERS`
reaches the server verbatim.

    ANTHROPIC_CUSTOM_HEADERS="x-tilerl-rollout: r42"
    -> x-tilerl-rollout: r42

So the tag is a header, not `metadata.user_id`. It costs one env var and one
`request.headers.get`, and the rows group without parsing a transcript. The
route now records `rollout` alongside the ids.

## No Docker, and what is actually available

The pod is already a container and cannot nest one; the Mac has no bubblewrap.
Isolation is Claude Code's own sandbox, configured through `--settings` (there
is no `--sandbox` flag; it is a settings key). Probed on both hosts:

| host | sandbox | probe |
|---|---|---|
| this Mac | Seatbelt | `/usr/bin/sandbox-exec` present |
| the pod | bubblewrap + userns | `unshare -Ur true` returns 0, uid 0; **`bwrap` is not installed** — `apt-get` is present and the pod has network (registry reachable, HTTP 200) |

`failIfUnavailable: true` is the key that matters: without it a host that
cannot sandbox runs the agent unconfined, and a rollout that touched the real
filesystem is worse than a rollout that did not happen.

## The sandbox test has a negative control, and it needed one

The obvious test — script an escape, assert the file is absent — passes just as
well when the command never ran at all. So the same scripted escape
(`echo pwned > <outside cwd>`) runs twice:

| run | file outside cwd | what it shows |
|---|---|---|
| `sandbox=True` | **absent** | the sandbox blocked the write |
| `sandbox=False` | **present** | the escape works, so the first result is real |

Both halves are in `test_sandbox_confines_writes_to_the_rollout_dir`. One
without the other measures nothing — this is the same lesson as stage 1's
unreachable gate, arriving from the other direction: there, a gate that could
never go green; here, one that could never go red.

## The scripted engine paid for itself

The whole launcher gate runs against `_ScriptedEngine`, so it needs no weights
and finishes in ~7s. That was the second use of a thing built for stage 1's
semantic assertion, and the reason it was worth building rather than stubbing
inline in one test.

Verified the CLI really runs rather than the assertions passing vacuously:
`rc 0`, `num_turns 1`, `result 'done: two files'`, one record row
(`request_id 1`, `stop_reason end_turn`, 16 completion ids, 16 logprobs).

## Known-wrong, and why it is not fixed here

tilerl-e4 read `/work/Qwen3.8-27B-NVFP4/chat_template.jinja` on the pod the same
day. The 27B does **not** emit `{"tool": ..., "input": ...}`; it emits
`<tool_call><function=NAME><parameter=P>…`, tool results come back as
`<tool_response>` (not `<tool_result>`), and the generation prompt carries
`<think>\n` rather than tileRL's bare `assistant\n`. All of that is real and all
of it is wrong in the shim today.

It is not fixed in this entry because `tiny` uses `ByteTokenizer`, which has no
`<tool_call>` token: hardcoding the 27B form would make both no-weights gates
meaningless, and those are the only ones that run every day here. The plan is to
dispatch on whether the tokenizer has the tool tokens — one predicate, not an
abstraction layer. Pending e4's answer on whether that is worth it or whether
tiny's gate should simply degrade.

Also open: `SamplingParams` has no `top_k`, and the 27B's card expects
temperature 1.0 / top_k 20 / top_p 0.95.

## Rule

A guard is only a guard if you have watched it refuse. Every isolation or
safety assertion ships with the negative control that would fail if the
mechanism were absent — in the same test, not in a comment.
