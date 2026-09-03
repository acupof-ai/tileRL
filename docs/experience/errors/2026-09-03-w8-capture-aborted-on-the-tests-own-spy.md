# The W=8 capture aborted on the test's own spy, and the fallback is engine-wide

## Context

The CUDA suite on latest main passes, and every run prints:

```
engine.py:753: UserWarning: decode graph capture failed for B=1 W=8 (Cannot copy
between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is
pinned.); eager fallback
```

Found by running the pod CUDA suite, which CI never runs — CI is
`TILERL_TARGET=cpu` on ubuntu and macos only, and the CPU target captures no
graphs, so this line has never appeared in CI output.

The warning names a width the spec work depends on. Running it down took
re-raising the swallowed exception on a scratch copy of the tree, because the
`except Exception` at `engine.py:750` keeps `exc` in the message and drops the
traceback.

## Root Cause

**The test's instrumentation aborts the capture it is instrumenting.**
`test_e2e.py::test_verify_commits_the_trunks_own_draw` wraps
`backend.gdn_decode` with a spy that calls

```python
written.update((int(s), p) for s in torch.as_tensor(slots).reshape(-1).tolist() ...)
```

`.tolist()` on a `cuda:0` tensor is a device-to-host copy, which CUDA graph
capture forbids. The capture aborts inside `_DecodeGraph.__init__` at
`model.forward -> _gdn -> gdn_decode`, on the spy, not on any engine code.

**Two things that make this worse than one aborted capture.**

The fallback is not scoped to the width that failed: `engine.py:756` sets
`self._decode_graph_on = False`, so one failure at W=8 turns capture off for the
engine, W=1 included, for the rest of its life.

And the gate could never have tested the captured path. Every assertion it makes
reads a Python spy — `w_scatter`, `w_decode`, `w_select`, `w_verify`. Those run
once while the graph is being recorded and never again at replay, since a replay
launches kernels and calls no Python. So a green run of this test says nothing
about the graph either way; it was an eager gate that did not say so.

## Fix

`decode_graph=False` on that engine, with the reason in a comment. The test's
behaviour is unchanged — it was already running eager, by way of a failed
capture — and the engine no longer attempts a capture that cannot succeed.

Verified on an H20: the full CUDA suite under
`-W "error:decode graph capture failed"` is `188 passed, 9 skipped`, where before
it raised on this test.

**Coverage this leaves open, stated rather than closed:** nothing now exercises
W=8 capture on CUDA. It cannot be a CPU test, and the previous state was not
coverage — it was a capture that aborted and a fallback that hid it. A CUDA-gated
gate that asserts `_decode_graphs` holds `(B, 8)` after a verify tick would be
real coverage, and does not exist.

## Rule

A spy that copies a tensor to the host cannot observe a captured path. If a test
asserts through Python callbacks, it is an eager test — build its engine with
`decode_graph=False` and say so, rather than letting the capture fail into the
same behaviour.

A fallback that warns and continues turns a defect into a line of log nobody
reads. When the fallback is also wider than the failure — here, capture off for
every width after one width failed — the warning has to say so, or the blast
radius is invisible at the point where someone decides to ignore it.
