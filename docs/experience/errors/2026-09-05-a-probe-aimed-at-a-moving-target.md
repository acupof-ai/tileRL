# A negative control aimed at HEAD~1, and a race the assertion shape decided

**Date:** 2026-09-05 · **Class:** error · **Where:**
`scripts/probe_ui_gate_negative_control.py`, `tests/test_server.py:984`

Two defects found in checks rather than in the code they check. Unrelated mechanisms, one
property in common: both were green or silent for reasons that have nothing to do with what
they assert.

## 1. A negative control whose target moves every commit

`probe_ui_gate_negative_control.py` proves the JS reference gate catches the real
`addThinking` bug, not just a synthetic string. It read `HEAD~1`'s source.

`HEAD~1` is relative. It named the broken commit on the day the probe was written and
something else on every day after, so the probe's useful life was **one commit**. Run
against main on 2026-09-05 it fails with *"the gate does NOT catch the bug it was written
for — it is decoration"* — a message about the gate, produced by the probe pointing
somewhere else. Nothing noticed because it lives in `scripts/` and CI never runs it.

Pinned to `69398d4`, verified before pinning rather than after:

| revision | `_CHAT_UI` leaves `addThinking` unresolved? |
|---|---|
| `69398d4` (parent of the fix `e954f8c`) | **yes** — probe's assertion holds |
| `e954f8c` (the fix) | no |
| `HEAD~1` today | no |
| `origin/main` | no |

Exactly one revision satisfies it, and it is the one the probe means. Runs green pinned.

**#90 fixed the symptom and left the defect.** The asset split moved `_CHAT_UI` to
`ui_assets.py`, and #90 taught the probe to look in both files — a real fix for the path,
which the split would otherwise have broken. `REV = "HEAD~1"` stayed. The path was what
the split broke; the ref was already broken, and a check reading the right file at the
wrong revision is no better off.

`scripts/baseline.py:74` also uses `HEAD~1` and is **not** the same defect: it checks that
any two adjacent commits order correctly, so a relative ref is the right semantics there.
Checked rather than assumed from the shape.

## 2. A count-based injection that raced the engine loop

`test_v1_messages_never_answers_200_for_an_engine_failure` wraps the engine in a `_Dies`
that lets one `take` through and raises on the next (`_left = 1`). When the request
finished before the route's first poll, that first `take` returned the completed tokens,
`_run` left its wait loop, and the raising call never happened — so the non-stream arm
answered **200** while the stream arm, whose allowance was already spent, answered **400**,
tripping the agreement assertion.

Measured, because "passes alone, fails in the suite" is the shape of a flake and not proof
of innocence: **1 of 12 runs, signature `non-stream 200 but stream 400`.** After keying the
injection on the *result* (`take` returns None → still running, pass it through; the first
one carrying a result raises): **0 of 12.**

**The same construction next to it does not flake, and the reason is the assertion, not the
injection.** `_DiesAfterOnePeek` uses the identical `_left = 1` pattern and measured **0 of
12** — because it asserts a *contract* (`[DONE]` or an explanation), and an injection that
never fires leaves a normally-terminated stream, which satisfies the contract. Left alone. A
count-based injection is only a race when the assertion can tell the difference between "the
failure was injected" and "the request succeeded".

## Rule

A check's target must be as pinned as its claim. A relative ref (`HEAD~1`) or a call count
(`the second call`) is a moving target: the first re-aims with every commit, the second with
every scheduling outcome. Pin the sha; key the injection on the state you mean rather than
on how many calls it took to get there. And when a check fails once and passes alone,
measure the rate before deciding it was noise — 1 in 12 is a defect with a signature, not
weather.
