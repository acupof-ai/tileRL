# Probe arity drift burned a V100 down-window — sm70, 2026-09-16

## Context

The H2 cmax-bucket sparse-graph probe (`scripts/probe_sparse_graph_cmax_bucket.py`,
untracked, prepared the same morning) was to run in a scheduled V100 window: stop the
hybrid serve, run the probe (~15 min), restart. The window opened, serve was stopped
cleanly (GPU verified 0 MiB), and the probe died on tick zero:

```
TypeError: build_engine() missing 1 required positional argument: 'backend'
```

Zero buckets executed. The live serve stayed down ~30 minutes while the script was
fixed and the serve brought back. The serve was restored at the same tree
(`1e9630dee4`, boot 0, idle 200 in 0.28 s).

## Root Cause

#599 moved engine assembly into `tilerl.build` and made `build_engine`'s signature
`build_engine(cfg, model, backend, *, ...)`. When the probe was first adapted to the
refactor, the stale imports (`_build_model` → `build_model`, `build_engine` module)
were renamed, but the call site kept passing `(model, be)` — missing the new leading
`cfg`.

Every pre-window check passed because none of them executes a call:

- `python -m py_compile` and `ruff check` prove syntax and resolvable names, not
  positional-argument contract;
- the CPU check that did run covered only the pure bucket-math helper, never the
  engine construction.

The probe was prepared once and then sat untracked while the deploy tree moved. A
waiting instrument is the worst case: nothing runs it between authoring and the
scheduled window, so drift is invisible until the device time it consumes.

## Fix

`cfg` is threaded through `_engine` / `probe_bucket` / `probe_b4_mmlu`; before the
next window the probe is loaded via `importlib` on CPU and every cross-module
function it calls is checked with `inspect.signature`, plus its object construction
is exercised on the CPU target.

## Rule

**Before opening a device down-window for a probe, bind the call, not the file:
import the probe locally and assert each cross-module call's arguments against
`inspect.signature` of the function in the tree being deployed.** Re-run the check
after every merge to that tree — a probe waiting for a scheduled window has no
execution between authoring and the window, so refactors move silently under it.
A syntax check answers "does this parse"; a down-window needs "does this construct".

Same family as [the eval cap measured itself](2026-09-04-the-eval-cap-measured-itself.md):
a green check that exercised the instrument's wrapper instead of the path the result
depends on.
