---
question: Why did three separate checks pass today while the thing they were supposed to guard was broken or untested?
status: measured
source: three instances in tileRL and aupai, all 2026-09-02, each caught by something other than the check itself
---

# A check constrains the cases it runs, and nothing else

Three times in one day a green check meant less than it looked like it meant.
Different mechanisms, same shape, so the shape is the lesson.

## The three

**1. The sandbox gate that could not go red.** `test_sandbox_confines_writes_to_the_rollout_dir`
scripts an escape (`echo pwned > <outside cwd>`) and asserts the file is absent
when sandboxed. That assertion passes just as well when the command never ran
at all. The fix was a negative control in the same test: run the identical
escape unsandboxed and assert the file **appears**.

It earned its place within the hour. Changing the tool-call format to the 27B's
XML invalidated the scripted escape, and the control is what failed —
"the escape did not write even unsandboxed, so the sandboxed assertion above
proves nothing". Without it, the sandbox test would have stayed green while
testing nothing at all.

**2. The selftest registered in the wrong worktree.** Adding
`scripts/trace_classes.py` and its `SELFTEST_FILES` entry in one commit was
refused by the hook: `.git/hooks/pre-commit` resolves against **main's**
worktree, so the running hook did not yet carry the path. AGENTS.md:382 records
the worse version of this — a file registered on a branch, every commit printing
`selftests 0.03s`, and the selftest never running once. A timing line cannot
tell you the run was empty.

Registered first, merged, then added the file. Then broke one assertion on
purpose and confirmed the hook refuses. **"The path is in the set" and "that
entry rejects things" are different claims.**

**3. The golden capture that proved only what it exercised.** PR #5 unified the
two prompt renderers behind byte-equal golden strings captured from both routes
before the change. Correct gate, correctly built, and it caught none of the
three real findings in review:

- the thinking default flipped (`_MAX_THINK.get(None)` is `None`, `None != 0`
  is True), so requests without `reasoning_effort` began opening a `<think>`
  block that the OpenAI route then returned to clients verbatim;
- `sampling()` accepted `max_think_tokens` and discarded it when
  `thinking=False`;
- OpenAI `image_url` parts vanished silently, because `blocks_to_text` knows
  Anthropic's block names only.

All three live in paths the captured strings never touch. The capture made a
silent reordering impossible, which is exactly what it was asked for. It was
never evidence of "zero behavior change."

## The shape

Each check was real, correctly built, and green. What varied was the distance
between **what it ran** and **what it was believed to prove**:

| check | ran | believed |
|---|---|---|
| sandbox test | one escape, sandboxed | the sandbox isolates |
| selftest registration | the files the hook knew | every selftest is gated |
| golden capture | four prompt shapes | zero behavior change |

The gap is invisible from inside a passing run. Nothing in a green result
reports the cases it did not cover.

## Rule

**Watch it refuse before trusting it.** For any guard, write the run where the
mechanism is absent and confirm the check fails — in the same test where that
is possible, by hand otherwise, before the check is credited with anything. A
guard you have never seen fail is a hypothesis about a guard.

And when reporting one: say what it exercised, not what it suggests. "The
golden capture is byte-equal on four shapes" is a fact. "No behavior change" is
a claim about every input, and no finite capture makes it.
