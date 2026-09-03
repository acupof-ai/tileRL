# `_quantize_draft` was not idempotent — V100 (sm70), 2026-09-02

> Status: fixed, gated. A second `build_engine` over the same draft head packed
> the already-packed `fc.wq` into `fc.wq.wq`, and the draft forward then raised
> `KeyError: 'fc'`. Nothing on the load path noticed.

## Context

Profiling wanted three engines over one model — dense, spec depth 1, spec depth 3
— to difference their launch counts. The second one died:

```
KeyError: 'fc'
```

from inside `DraftHead.forward`, at `self.layers._linear(backend, …, "fc")`.

## Root Cause

`build_engine` serves the draft's weights and writes them back **in place**,
because `DraftHead.layers` is a `Model` holding that same dict:

```python
served = backend.materialize(_quantize_draft(draft.params, fp4=fp4))
draft.params.clear()
draft.params.update(served)
```

`_quantize_draft` packs every 2-D tensor at least 128×128. After one pass `fc` is
gone and `fc.wq` is in its place — which is 2-D, 256×256 after packing, and over
the threshold. So the second pass packs it again into `fc.wq.wq`, and the plain
`fc` lookup has nothing to find.

The failure is loud when it happens but the *cause* is silent: nothing between
loading a draft and using it asks whether the weights are already served, and the
error names `fc`, which is present in the checkpoint and correct on disk.

## Fix

An early return when the dict is already served:

```python
if any(k.endswith((".wq", ".w8")) for k in params):
    return dict(params)  # already served
```

## Why this was worth fixing rather than working around

The script could have loaded the draft after the dense case — that is a two-line
change and I wrote it first. But the same shape reaches real code paths: a train
loop that rebuilds an engine per phase, an A/B that compares configurations, the
roadmap's recapture-after-update. One engine per process is the shipped path
*today*, which is exactly the kind of assumption that stops being true without
anyone revisiting the function that depends on it.

## Rule

**An in-place transform of shared state has to be idempotent, or say so.**
`_quantize_draft` reads a dict and its caller writes the result back over the
input. That is a fixed point by construction or a bug waiting for a second call;
there is no third option, and which one it is should be decided in the function,
not by how many times its caller happens to run.

Second: **a KeyError names what is missing, not what removed it.** `fc` is in the
checkpoint, passes the loader's own `missing` check, and is correct on disk. The
useful question was not "where is fc" but "what else could have consumed it",
which points at the only writer of that dict.

## Gate

`tests/test_draft_loader.py::test_quantize_draft_is_idempotent` — packs a dict
twice and asserts the key set is unchanged and no `fc.wq.wq` appears. Negative
control: removing the early return fails it with `AssertionError: a second pass
must be a no-op, not a re-pack — 'fc.wq.wq'`.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-02 | (this change) | any | any | draft head | second `build_engine` | **works** |
| 2026-09-02 | a61ffc1 | any | any | draft head | second `build_engine` | `KeyError: 'fc'` |

No runtime effect on the shipped single-engine path: the early return only fires
on a dict that has already been served, which cannot happen there.
