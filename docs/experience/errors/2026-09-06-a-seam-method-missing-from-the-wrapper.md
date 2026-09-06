# A method added to a seam is missing from the wrapper — server, 2026-09-06

> Status: closed by the gate in this entry

## Context

`serve --devices` puts `DataParallelEngine` where the routes expect an `Engine`. Twice now a
name the routes call has been present on `Engine` and absent from the wrapper, and both times
it shipped and broke every request on that path:

| date | name | symptom under `--devices` |
|---|---|---|
| earlier | `limits` | **400** on every Claude Code turn — `messages.py`'s `getattr(engine, "limits", None)` read `None`, the clamp went dead, and `max_tokens=32000` reached `submit` unclamped |
| 2026-09-06 | `room_for` | **500** on every request that omitted `max_tokens` |

The second one is mine. Adding `room_for` to `engine.py` for the omitted-cap default made it a
requirement on every implementation of the seam, and I only noticed because 16 SDK tests went
red — those failures were about test *doubles*, not about the wrapper.

## Root Cause

**The failure the doubles reported and the failure that mattered were different failures.**
`AttributeError: '_PromptKeyedEngine' object has no attribute 'room_for'` names a double in
`tests/test_api_sdk.py`. It is loud, it is harmless, and the two obvious ways to silence it
both leave the real defect shipping:

- `getattr(engine, "room_for", <fallback>)` in the route — turns the 500 into a silently wrong
  cap, and hides the requirement from grep.
- give each double a `room_for` — correct, and says nothing about `DataParallelEngine`, which
  is not a double and is not in that file.

I took the second path and then asked which *non-test* implementations exist. That question is
what found the 500. Nothing in the red output pointed at it.

**Why one arm per method does not catch the next one.** The `limits` defect got a test:
`test_the_clamp_survives_the_data_parallel_wrapper`, plain vs wrapped, both must agree. It is a
good test and it passed throughout the `room_for` defect, because it exercises `limits`. A
per-method arm catches the method it was written for. The seam has 8 names.

## Fix

`parallel.py:95-98` forwards it, `min` across replicas for the same reason `limits` takes the
min: `submit` routes to the shortest queue, so the only budget every replica honours is the
smallest one's.

The gate is `test_every_engine_the_routes_accept_implements_what_they_call`. It does not carry
a list of methods — it derives the required set from the route modules' own source and checks
it against every implementation:

```python
for mod in (server, messages, responses):
    src = inspect.getsource(mod)
    called |= set(re.findall(r"\bengine\.([a-z_][a-z0-9_]*)", src))
    called |= set(re.findall(r'getattr\(\s*engine\s*,\s*"([a-z_][a-z0-9_]*)"', src))
```

So a route that starts calling `engine.foo()` extends the requirement with nobody remembering
to add an arm. Today the set is `submit take peek stop_text logprobs stats room_for limits`.

**Two controls, both red for the reason under test:**

- drop `room_for` from the wrapper → `DataParallelEngine is accepted by the routes but does not
  implement ['room_for']`.
- add a call to a method that exists nowhere → `Engine ... does not implement
  ['brand_new_seam_call']`. This is the control that matters: it shows the gate catches the
  class, not the two instances already known.

## Two ways this gate was wrong before it was right

**`hasattr(Engine, "limits")` is False.** `limits` is assigned in `__init__`, so a class-level
probe reports it missing on a completely correct engine — and `DataParallelEngine` declares it
as a `@property`, so the class-level probe *passes* there. The first version of the gate
therefore failed on the correct implementation and passed on the one with the history of
missing attributes, which is exactly backwards. It builds instances now.

**The regex read a comment as a call.** It demanded `_thread`, which appears only in
`server.py:236`, inside a comment explaining why `/health` deliberately does **not** check loop
liveness — `DataParallelEngine` has no thread, so the check would pass vacuously. Prose naming
an attribute the code refuses to use is textually identical to a call. It also matched
`engine.py` from a docstring. Both are excluded by name with the reason written down.

The derived set is asserted non-empty and to contain the 8 expected names, because a regex that
silently matched nothing would make every implementation pass — the state where this gate is
worse than no gate.

## Still open on this seam

`/health` cannot check loop liveness under `--devices` at all: the only signal is
`Engine._thread` and the wrapper has no equivalent, so health reports `ok` from `stats()` alone
even with every replica's loop dead (`server.py:234-238` records this). Not fixed here; it
wants a liveness method on the seam. That would be the third thing the wrapper cannot answer.

## Rule

**When a red test names a test double, ask which non-test implementations exist.** The loud
failure and the shipping failure were in different files, and the fix that silenced the loud
one would have left the other in place. A double's missing method is a signal that the seam
widened, not the defect itself.

**A gate for "every implementation has every method" must enumerate both, and derive at least
one of them.** A hardcoded method list is a per-method arm with extra steps: it goes stale the
first time a route calls something new, in exactly the situation the gate exists for.

**Probe the object the caller holds.** Callers hold instances, so `hasattr` on the class
answers a different question — and answers it inconsistently across implementations, depending
on whether each sets the attribute in `__init__` or declares it as a property.
