# A text check cannot see whether the line runs — twice in one night, two sessions

## Context

Two sessions independently shipped an assert that greps source text for a string, and both
asserts passed while the behaviour they were written to protect was broken.

**This session,** `scripts/steps_to_reward.py`. The gsm8k branch had called
`get_tokenizer(a.model)`, which 401s — the model name resolves only through
`cli._qwen38_tokenizer`. The fix was one line; the assert protecting it was

```python
assert "_qwen38_tokenizer" in inspect.getsource(make_reward)
```

Mutating the call back to `get_tokenizer(a.model)` **passed**. The comment written above the
call to explain the fix contains the string `_qwen38_tokenizer`, so the grep was satisfied by
the explanation of the fix rather than the fix.

**tilerl-0a, same night,** `math_jsonl.py`: `assert '+ _INSTRUCTION' in text` — which sees the
text of a line, not whether that line can execute.

## Root cause

Both asserts test the *presence of a string in the source*. Every mechanism that can break
the behaviour while preserving the string then passes: a comment holding the same name, a
line that is present but unreachable, a name bound but never called, a duplicate definition
shadowing it. The source is the input to the behaviour, not the behaviour.

The failure is specifically permissive. A grep-based assert almost never produces a false
alarm — it produces a false pass, and only when something is actually wrong, because a
correct implementation contains the string too. So it is green in exactly the case it cannot
distinguish, and its green is the same green as a working check's.

Writing the explanation next to the fix made this worse rather than better: the comment
density the tree asks for is what satisfied the probe.

## Fix

Check what ran, not what is written. Here: monkeypatch both candidate functions to append a
tag to a list, call `make_reward`, and assert the list.

```python
called = []
_cli._qwen38_tokenizer = lambda: (called.append("by-name"), ByteTokenizer())[1]
_tokmod.get_tokenizer = lambda src=None: (called.append(f"hub:{src}"), ByteTokenizer())[1]
...
assert called == ["by-name"], f"...got {called}"
```

The mutant now dies with `got ['hub:qwen38-27b']` — it names which wrong function ran, which
a grep-based assert could not have said even when it failed.

## Rule

An assertion over source text is not a test of behaviour. When the thing to protect is "this
call goes to X", record the call; when it is "this line executes", make the execution
observable. A probe that a comment can satisfy is a probe that comment density will
eventually satisfy — and the tree asks for comments next to exactly the lines worth
protecting.

Related: [the same run's other two defects](2026-09-08-a-gate-green-produced-by-no-gradient.md),
and the same night's `math_jsonl.py` instance found by tilerl-0a.
