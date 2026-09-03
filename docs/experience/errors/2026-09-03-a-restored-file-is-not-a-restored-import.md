# A restored file is not a restored import — 2026-09-03

## Context

Negative controls here are run by mutating a source file, running the gate, and
moving a `.bak` back:

```
open(p+".bak","w").write(s); open(p,"w").write(mutated)   # mutate
pytest -k gate                                            # expect red
mv p.bak p                                                # restore
```

Ran that on `engine.py` to check a new gate discriminates. It did. Then the full
suite came back **1 failed** on the very gate that had just passed, and the failure
was the mutation's own signature — `widths [1,2,3,4,5]` against `_width 4`.

## Root Cause

The `.pyc` was stale. CPython invalidates bytecode on **(source mtime, size)**, and
the mutation and the restore landed inside the same mtime tick with the same file
size — the mutated bytes and the original differ by one character, `1 + self._width`
vs `2 + self._width`. So the cache looked valid and the interpreter kept serving the
mutated bytecode from a file whose bytes on disk were correct.

Every direct check agreed the source was right, which is what made this expensive:
`grep` found one `1 + self._width` and zero `2 + self._width`; `inspect.getsource`
printed the correct line (it reads the file, not the code object); `e._width` was 4
and `range(1, 1+4)` was `[1,2,3,4]`; and there was no override of the method
anywhere in the tree. I went looking for a runtime monkeypatch that did not exist.

What settled it was `dis.dis(m.Engine.graph_keys)` — the only tool that reads the
loaded code object rather than the file:

```
32 LOAD_CONST  2 (1)
34 LOAD_CONST  3 (2)      <- the mutation, in the bytecode, not on disk
38 LOAD_ATTR   2 (_width)
48 BINARY_OP   0 (+)
```

## Fix

`find . -name __pycache__ -prune -exec rm -rf {} +`, then the suite passed 235.

The durable fix is in how the control is run: **delete the bytecode as part of the
restore**, not as a recovery step.

```
mv p.bak p && find . -name '__pycache__' -prune -exec rm -rf {} +
```

Or avoid writing the file at all — `monkeypatch.setattr` on the attribute under
test mutates the loaded object directly and leaves no cache to go stale. That is
the better shape for a control on a Python-level value; file mutation is only
needed when the gate parses source text.

## Rule

When a source file and its behaviour disagree, suspect the bytecode cache before
suspecting the code — and confirm with `dis`, not with `grep` or
`inspect.getsource`, both of which read the file the interpreter is ignoring. A
one-character edit reverted inside the same second is invisible to `.pyc`
invalidation, which keys on mtime and size, so the class of edit most likely to go
stale is exactly the negative control: minimal, and immediately undone.
