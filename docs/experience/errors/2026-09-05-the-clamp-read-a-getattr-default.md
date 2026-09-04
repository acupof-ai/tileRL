# `serve --devices` 400-ed every Claude Code turn, because the clamp read a default

**Date:** 2026-09-05 · **Class:** error · **Where:** `src/tilerl/messages.py:153`,
`src/tilerl/parallel.py`

## Context

`/v1/messages` has to accept `max_tokens: 32000` — Claude Code sends that on every
turn, the real API accepts it and stops at the context edge, and `Engine.submit`
refuses `prompt + max_new_tokens` over `max_total_tokens`. So `mount_messages`
clamps:

```python
engine_limit = getattr(getattr(engine, "limits", None), "max_total_tokens", 0)
...
budget = max(1, engine_limit - len(input_ids)) if engine_limit else req.max_tokens
```

`DataParallelEngine` forwards 11 methods explicitly and has no `__getattr__`. It had
no `limits`.

## Root cause

The two `getattr` defaults turned a missing attribute into a limit of **0**, and
`if engine_limit` reads 0 as "no limit known" — so the clamp fell through to
`req.max_tokens`, the request went to the engine at its asked-for 32000, and
`engine.py:459` rejected it. The 400 is correct behaviour from the engine; the defect
is that the clamp in front of it went silent.

Measured, one engine, one request, `max_total_tokens=512`, only the wrapper differing:

| wrapper | `engine_limit` | POST /v1/messages |
|---|---|---|
| plain `Engine` | 512 | **200** |
| `DataParallelEngine([e], [cpu])` | 0 | **400** `request (32060 tokens) exceeds max_total_tokens (512)` |

So `serve --devices` answered 400 to every Claude Code turn while the same engine
unwrapped answered 200.

This is the third instance of one shape: **code doing an existence check against a
duck type.** `peek` was the first (#80: `AttributeError` past the SSE 200 header → 200
with zero frames), the `_thread` liveness check I nearly added to `/health` was the
second (#83, blocked before it landed — it would have passed vacuously under
`--devices`). A `getattr` default is the quiet variant: the first two raised, this one
returned a number that reads as valid.

## Fix

A `limits` property on the wrapper, not a third default in `messages.py`:

```python
@property
def limits(self) -> Any:
    return min((e.limits for e in self._engines), key=lambda lim: lim.max_total_tokens)
```

`min`, not `[0]`: `submit` routes to the shortest queue, so the budget a caller can
rely on is whatever the smallest replica accepts. `limits` belongs on the seam anyway —
`StepLimits` is half of the documented `submit`/`poll` + `StepLimits` contract, and a
caller clamping against it is the contract working.

Fixed on the wrapper because the alternative — a third `getattr` default, or
`hasattr` — treats the symptom and leaves the next reader of the seam to find the same
hole. `_serve_draft`'s reasoning applies: one definition, or a second thing to keep in
step.

## Gate

`test_the_clamp_survives_the_data_parallel_wrapper` runs the same request through both
wrappers and asserts **the two agree**, then that the agreed answer is 200. Two
assertions, because a fix that broke the clamp for everyone would make a one-arm test
pass by making both 400.

Negative control: with the property removed and `__pycache__` cleared, the gate reports
`plain 200, wrapped 400` and **fails**; restored, it passes.

## Surface, so this is the last one here

Every engine attribute the server layer reads, by grep over `server.py` and
`messages.py`: `submit`, `take`, `peek`, `logprobs`, `stats`, `limits`, `_thread`. The
last is only in a comment — the one I wrote in #83 saying it would be vacuous on DP —
so there is no live read. `limits` was the last of the six that `DataParallelEngine`
lacked; all six are now forwarded or defined.

**Not claimed:** that the other wrapper-shaped gaps in the tree are enumerated. This
covers the `server.py`/`messages.py` reads only. `train.py:213` reads
`engine._decode_graph_on` and `engine._prefix` off the engine and would raise
`AttributeError` on a `DataParallelEngine` — that path is not reachable today (training
engines are built unwrapped) and is not fixed here.

## Rule

`getattr(obj, "x", 0)` cannot distinguish "the attribute is missing" from "the
attribute is 0" — and here those two mean opposite things, because `if engine_limit`
treats 0 as *no limit to clamp against*, which is the most permissive reading, not the
conservative one. A numeric default silently picks a side of that ambiguity. `None`
would have raised on the next line.

So: where the attribute is part of a seam contract, define it on every implementation
of the seam rather than defaulting it at the call site. Where a default is genuinely
right, it has to be a value that changes nothing — and for a limit, 0 is the value that
turns the check off.
