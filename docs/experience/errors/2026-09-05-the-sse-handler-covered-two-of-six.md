# The SSE handler covered two exception types out of six, and 200 was the status for all of them — 2026-09-05

**Date:** 2026-09-05
**Arch:** target-independent (HTTP/SSE behaviour); measured on the cpu target
**Task:** local work, no GPU window (the resident 27B server holds 23.2 of 32 GB)

## Context

`DataParallelEngine` has no `peek`, and `_stream` calls `engine.peek(request_id)`
unconditionally — so `serve --devices 0,1` broke on every streaming request. What the
client saw was not an error: **HTTP 200, zero SSE frames, empty body**. That specific
defect is a peer's to fix. This entry is about why its symptom was a success status.

`_stream`'s only handler is `except (TimeoutError, RuntimeError)`. The 200 header and
`text/event-stream` content type are sent before the generator body runs, so anything
else it raises escapes after the response has already been committed as a success.

## Measured, before the fix

Six exception types injected at the engine boundary, `peek` and `take` both wrapped
(the two routes reach the engine through different methods, so injecting into one
leaves the other column blank). First call succeeds, second raises — the shape of a
request that dies mid-decode.

| exception | non-stream | stream | frames | error frame | `[DONE]` |
|---|---:|---:|---:|---|---|
| `TimeoutError` | **504** | 200 | 3 | yes | yes |
| `RuntimeError` | **500** | 200 | 3 | yes | yes |
| `ValueError` | **500** | 200 | **0** | no | no |
| `AttributeError` | **500** | 200 | **0** | no | no |
| `KeyError` | **500** | 200 | **0** | no | no |
| `IndexError` | **500** | 200 | **0** | no | no |

The non-stream route classifies all six. The stream route returns 200 for all six, and
for four of them delivers nothing at all — no content, no error, no terminator. A
client cannot distinguish that from a model that chose to say nothing.

### After the fix, same probe

| exception | stream | frames | error frame | `[DONE]` |
|---|---:|---:|---|---|
| all six | 200 | **3** | yes | yes |

Every row now behaves like the two that were already caught. The status stays 200 — it
has to, the header left long before — so the error frame and the terminator are the whole
signal, which is why the gate asserts those rather than a code.

**`#80` and this fix are independent, and that is checked rather than asserted.** I first
wrote "the table is unchanged by `#80`" having only run it once, after both landed. Removing
only the catch-all from a tree that carries `#80` reproduces the before-table exactly —
`ValueError`/`AttributeError`/`KeyError`/`IndexError` back to **0 frames, no error frame, no
`[DONE]`** — so `peek` was an instance and the handler set is the container. Restoring it
gives all six at 3 frames again.

## An intermediate result I nearly reported as a mechanism

The first version of the probe injected only into `peek`, so the non-stream column read
200 for every row. My hypothesis was that `__getattr__` on the wrapper was absorbing
`AttributeError` — plausible, and it would have been a tidy mechanism to write down.

Separating it took one more probe: the same exception through a wrapper **with**
`__getattr__` and through one where every method is bound explicitly. Both returned
**500**. So the hypothesis was wrong, and the real reason was duller — the non-stream
path polls `take`, which returned the finished request on its *first* call, so the
raising second call never happened. That column's 200 meant "the exception never
occurred", not "the exception was swallowed".

Injection timing changed the answer three ways, and only one of them describes the
question: `after=0` (raise immediately) gives **503 on both routes**, because the
failure lands in `submit` before streaming begins and the route's own handler catches
it; `after=1` gives the table above; and injecting into `peek` alone gives a
non-stream column with no information in it.

## Fix

A catch-all that frames the failure and logs it:

```python
except Exception as exc:
    logging.exception("stream for request %s died", request_id)
    yield _sse({"error": {"message": f"{type(exc).__name__}: {exc}",
                          "type": "internal_error"}})
    yield "data: [DONE]\n\n"
    return
```

**The log line is not decoration, and I checked it rather than asserting it.** A tidy
error frame is easier to ignore than silence, so a catch-all can make a
`peek`-class defect *harder* to find than the empty 200 it replaces. Measured with a
capturing handler: 1 record at ERROR, with `exc_info` present and `AttributeError`
named, alongside 3 frames + error frame + `[DONE]` to the client.

## The gate

`tests/test_server.py::test_a_stream_that_dies_mid_decode_does_not_look_like_success`
asserts the **contract**, not the status code: a stream either terminates with `[DONE]`
or says why it stopped. Written to be independent of *how* that holds, so a root-cause
fix elsewhere satisfies it too and this catch-all is not the only thing that can.

Negative control, two arms, because one of them is the interesting one:

| arm | gate | expected |
|---|---|---|
| catch-all removed (the bug as found) | **FAILED** | fail |
| only the `logging.exception` line removed | **passed** | pass |

The second arm is what proves the gate tests one thing. A gate that also fired on the
logging change would block an unrelated edit while claiming to be about the client
contract.

282 passed / 14 skipped, ruff clean.

## `/v1/messages` does not share the gap, and this is measured rather than read

I first wrote that as a code-reading claim. Running the same six injections through
`mount_messages`, both `stream=false` and `stream=true`:

| exception | non-stream | stream | events | error body |
|---|---:|---:|---:|---|
| `TimeoutError` | 503 | **503** | 0 | yes |
| `RuntimeError` | 503 | **503** | 0 | yes |
| `ValueError` | 400 | **400** | 0 | yes |
| `AttributeError` | 500 | **500** | 0 | no |
| `KeyError` | 500 | **500** | 0 | no |
| `IndexError` | 500 | **500** | 0 | no |

The two columns are identical, and no row is a 200. That is the structural reason
holding: `sse()` yields nothing until `_run` has produced the whole body, so every
engine exception happens before a header goes out and the route's handler still owns the
status code. Three of the six land on FastAPI's bare 500 rather than a typed error body,
which is a smaller and different problem — a client gets a status it can act on.

The contrast with the table above is the whole point of this entry: same engine, same
exceptions, and the route that streams *incrementally* is the one where a defect becomes
a success.

`test_v1_messages_never_answers_200_for_an_engine_failure` holds it. That gate is green
on the current tree, so on its own it would only record an observation — the negative
control is what makes it a guard, and the first arm is the interesting one:

| arm | gate | expected |
|---|---|---|
| `sse()` hoisted to yield **before** `_run` produces the body — i.e. the per-token rewrite the ponytail marker warns about | **FAILED** | fail |
| `ValueError` handler removed, so 400 becomes a bare 500 on **both** paths | **passed** | pass |

So the gate fires on exactly the change that would introduce this defect, and stays quiet
for a change that alters the status code without breaking the contract. It asserts the two
columns agree and that neither is a 200 — not a specific code.

## Not claimed

That any of the four uncaught types has been seen in production other than
`AttributeError` (the `peek` gap, measured today).

That the three bare-500 rows in `/v1/messages` are fine as they are. They are not a
silent success, so they are out of scope here, but a typed `internal_error` body would be
better than FastAPI's default.

## The same shape, one endpoint over: `/health` said ok for a wedged engine

Found while enumerating `server.py`'s handlers for the table above. `status` was the
literal `"ok"`, and `stats` fell back to `None` on any exception:

| engine state | before | after |
|---|---|---|
| healthy | `ok`, stats present | `ok`, stats present |
| `stats()` raises | **`ok`**, stats null | `degraded`, stats null, error named |

A health endpoint that cannot report ill health. Found by reading rather than by an
outage, and it had **no live consumer**: all three script readers take `stats`, not
`status`, and each already fails on `None` — `bench_workloads.py:127` maps it to `{}`
which trips its own `d["finished"] != 2` guard, and `probe_served_rate.py:56` raises
`TypeError` subscripting `None`. So neither can turn a broken engine into a fabricated
measurement. `scripts/serve_v100.sh` does not poll `/health` at all; it restarts on
process exit.

**But this one differs from the other two in who it deceives.** A 200 with an empty body
is wrong in a way a person notices — someone watching a reply appear sees nothing appear.
`status: "ok"` is written for a liveness probe, a load balancer, a restart policy. Those
do not find anything odd; they decline to restart a dead process. Of the three instances
in this entry it is the only one whose fix changes what a machine does, which is why it
is worth fixing while it still has no consumer rather than after one is added.

**Loop liveness is deliberately still unchecked, and the reason generalizes.** A third
state exists — the daemon loop dead while `stats()` answers fine from the request queues —
and the obvious check is `engine._thread.is_alive()`. I wrote it, then took it out:
`DataParallelEngine` has no `_thread`, so under `--devices` a
`getattr(engine, "_thread", None)` check passes **vacuously**. That is the exact shape of
the `peek` gap, written while fixing the class the `peek` gap belongs to — and written in
the defensive-looking idiom, which is what made it invisible.

So the root of this defect class is not "someone forgot to add a method". It is **code
doing existence checks against a duck type**: `hasattr`, `getattr(..., default)`, and an
unconditional call are three points on the same line, and only the last one fails loudly.
`DataParallelEngine` being a hand-written forwarding layer makes it likelier, not
different in kind. The fix that ends the class is to ask the interface a question
(`is_loop_alive()`) rather than ask the object whether an attribute exists — then a
forwarding layer that omits it fails at the call, which is the one form that gets noticed.

It also needs a `BaseException` to escape `_loop`'s `except Exception:
traceback.print_exc()`, so `MemoryError` rather than torch's OOM. The comment in
`server.py` records the rejected version and why.

Gated by `test_health_says_degraded_when_the_engine_raises`, which asserts the two states
**differ** and that the healthy one still says `ok` — so a fix that marks everything
degraded does not pass.

## Rule

A handler list on a generator that is already streaming is not the same as a handler
list on a function that returns a response. In the first case an uncaught exception
cannot change the status code, because the status code is gone — so the default
outcome of a defect is *success with no content*. Enumerate what escapes, and gate the
contract rather than the handler.

And when the reason a second path is safe is structural, measure it rather than reading
it: the structure is what makes the claim true today, and only a measurement says whether
the structure is actually what ships.

A third, from the check I nearly added: **an existence check against a duck type is a
silent pass waiting for a second implementation.** `getattr(obj, "_x", None)` and
`hasattr` read as careful and behave as skips; an unconditional call reads as careless and
fails loudly. When two classes implement one interface, ask the interface a question rather
than the object what it has.
