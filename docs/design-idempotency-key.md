# Design: client-generated idempotency key (retry reattachment)

Status: design, not scheduled. Depends on the server shape after #672 (the
`_drain_body` / `ws_chat` paths changed there); not queued in any current sprint.
Read-only proposal — no code here.

## Problem

A disrupted turn has no client-controlled identity. `/ws/chat` and the three REST
routes allocate a server numeric `request_id` inside `Engine.submit`
(`engine.py:827`). When the terminal frame is lost, or the server is unreachable
during a partition, the client's only recovery is Retry → a brand-new
`submit` → a second generation: double token cost and duplicated tool side
effects. The client cannot cancel or re-read a request whose id it never
reliably learned. See #677 for the two trigger sequences; #667 (cancellation
reachability) is related but distinct.

## Goal

A retry of the **same logical submission** reattaches to the in-flight or
recently-finished run instead of regenerating. Retry of an explicitly different
submission still runs normally.

## Key transport: body field, not a header

Add one optional field to every request body:

- WS ask JSON, `ChatCompletionRequest`, `MessagesRequest`, `ResponsesRequest`:
  `idempotency_key: str | None`.

Why body over an HTTP header: WS frames carry HTTP headers only once at
handshake, so a header cannot express a per-ask key on the playground's
multiplexed-per-socket transport; a body field is identical across all four
entry points. OpenAI has no standard for this (Stripe uses an
`Idempotency-Key` header; Anthropic has none), so we name it explicitly and
document it as an extension. The key is opaque to the model and never rendered.
A client that omits it gets today's behaviour (no dedupe).

Client rule (one line in `web/`): generate the key per user submission (random
UUID); Retry of that same turn reuses it; a freshly typed message gets a new one.

## Server state: a key registry in front of `submit`

A small module-level registry owned by the engine, populated and consulted
inside `Engine.submit` (under `self._lock`, next to `_next_id`, `engine.py:827`):

```
key -> {state: "running" | "done" | "failed",
        request_id: int,
        output: list[int] | None,     # held only for the post-completion window
        error: str | None,
        born: float}
```

Semantics on `submit(..., idempotency_key=k)`:

1. No key or key absent → current path, new `request_id`.
2. Key exists and `running` → return the **same `request_id`** (reattach); do not
   enqueue a second `_Req`, do not allocate blocks again.
3. Key exists and `done` within the retention window → return the stored
   `request_id` whose output a waiter can retrieve immediately (see the pop-once
   problem below).
4. Key exists and `failed` → raise the stored error (same as a fresh refusal so
   the client sees the honest failure, not a silent duplicate run).
5. Unknown key → run, and register it.

Retention/LRU — **byte budget first, entry count second**. The retained payload
is a completion's token ids, and a few large outputs can dwarf many small ones, so
capping entries alone (N × large output) is not bounded in memory. Use a total
retained-copy budget (~64 MiB) with an entry-count cap (~256) as a secondary
bound; evict oldest-first once either is exceeded.

The two states have different memory needs and get separate budgets:

- `running` entries store **only the `request_id`** (no output copy), so dedupe
  of in-flight retries costs almost nothing and is not subject to the byte cap.
- `done` entries are what hold the replayable output copy and draw on the 64 MiB
  budget. A single completion larger than a per-entry ceiling (e.g. 16 MiB) is
  not retained after it finishes — the key still deduped while running, but a
  post-completion attach after such a giant run degrades to a fresh run rather
  than pinning an outsized buffer.

Eviction means a late retry re-runs, which is the correct degradation (the
promise is best-effort within the window, not exactly-once forever). Age remains
a coarse tie-breaker; the byte budget, not a fixed minute count, is the primary
bound.

### The pop-once constraint (the one real design wrinkle)

`Engine.take` pops `_finished[rid]` (`engine.py:947`), and `await_completion`
loops on it (`prompt.py:367`). A completion can therefore be consumed by exactly
one waiter — a second connection attaching after the first finished gets
`None`/KeyError today. The registry must not route two waiters at the same
poppable row.

Two viable shapes:

- **(Preferred) registry holds the output independently of `_finished`.** On
  completion, store the output token list in the key entry; an attach after
  completion reads the registry copy (a per-attach copy, not a pop), while the
  normal `_finished` pop still feeds the original waiter. Reference-count the
  row: keep `_finished`/KV release behavior driven by the engine as today, and
  free the registry copy at LRU eviction. This decouples "result retained for
  replay" from "row's KV lifetime".
- (Alternative) make completion storage multi-subscriber (replace the pop with
  a read + subscriber count). More invasive — touches the hot `take` path and
  every existing waiter; not worth it for one feature.

## Delivering one result across the three shapes

The registry returns a `request_id`; each transport then behaves as if it had
submitted that id:

- **non-stream (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`)**:
  run the existing `await_or_cancel`/`await_completion` wait on the returned id.
  If it already finished, the first take returns immediately; an attaching retry
  reads from the registry copy. Same response shaping, no special path.
- **SSE (`stream:true` / responses SSE)**: re-drive `_deltas` for the attached
  id. Two sub-cases:
  - still running: a second SSE consumer iterating the same id would contend for
    the pop-once `peek/take` tokens. Minimum safe behaviour for a **retry** is to
    not promise live-mid-stream fan-out: if the run is still in flight, the
    attaching stream reads the current prefix via `peek` (lock-free,
    `engine.py:925`) and then waits for completion, delivering prefix+remainder;
    it does not double-consume. (True multi-fan-out of the live token stream is
    out of scope; a retry is rare and a prefix-replay + tail is sufficient and
    simple.)
  - finished: emit the stored output as one normal completed stream.
- **WS `/ws/chat`**: same as SSE — on `ask` with a key that resolves to a
  running id, the handler re-enters the `_deltas` poll loop for that id and
  streams prefix-onward; finished → replay once.

## WS reconnect after the old socket died

This is the playground case. Sequence:

1. Turn submitted with key `k`; socket drops mid-stream.
2. Server keeps the run (its lifecycle is independent of the socket; the
   disconnect-cancel path is what #672 fixes — interaction below).
3. Client Retry opens a new socket, sends the same ask **with `k`**.
4. `submit` with `k` returns the existing `request_id`; handler reattaches,
   `peek`s the prefix, streams to completion. No second generation.

The engine does not need to know about sockets at all; key→request_id is the
only shared state.

## Interaction with #672 (cancel) — attach vs cancel on retry

The decision a Retry implies for the old run:

- **Attach (recommended).** Retry means "I still want this answer"; reusing the
  key attaches, and the old request keeps running exactly once. This is what the
  registry above does. It matches user intent (Retry after a *connection* loss is
  not "I changed my mind") and it is the only choice that avoids duplicate work.
  Stop, not Retry, is the cancellation gesture: the WS close on Stop carries no
  key reuse (the client discards the key for that turn), so #672's cancel path
  is unaffected.
- Cancel-and-rerun: a Retry that first cancels the old id then submits anew
  would reintroduce a generation (and, pre-#672, a possible ~80s prefill hold),
  defeating the purpose; it also cannot guarantee the two runs differ. Reject
  this semantic.

Net: **Retry = attach (idempotent), Stop = cancel (key discarded).** The client
must never reuse a key after the user pressed Stop. State that explicitly in
`web/README.md`.

One race to name: a Retry that arrives while an earlier Stop's cancel is in
flight (pre-#672, prefill not yet preempted) — same key, one path wants cancel,
the other attach. Resolution: cancel is terminal on the key; once a key is marked
cancelled/failed it is never reattached, so an attach after a Stop returns the
stored failure rather than resurrecting the run. This makes the ordering
unambiguous without depending on #672's timing.

**Lock boundary for that race: attach and cancel must be serialized under the
engine lock.** Every registry state transition — running→done, running→failed,
running→cancelled, and the read-reattach on submit — has to happen inside
`Engine._lock` (the same lock `submit` and `cancel` already take). If an attach
read the key outside the lock, a concurrent `cancel` could flip it to cancelled
between the read and the reattach and the retry would still bind to a row being
abandoned. Submit already consults the registry under `_lock` (above) and
`cancel` (`engine.py:2429`) takes it too; the requirement is simply that no
key-state lookup or flip is added outside that critical section. No new lock.

## Files and minimal steps (server; client is one field + one rule)

- `src/tilerl/engine.py`
  - add the key registry (dict + LRU deque/timestamps) near `submit` (~`:627`,
    consulted/set under `_lock` in `submit` ~`:827`).
  - record terminal state on the existing finish/fail paths (`_finish` ~`:2457`,
    `cancel` ~`:2429`, the `_failed` path) and retain an output copy for the
    replay window.
  - add a `submit`-adjacent resolver: key → existing/new `request_id`, or raise
    stored failure; keep bare `submit(input_ids, params)` unchanged.
- request models: `ChatCompletionRequest` (server.py ~`:80`),
  `MessagesRequest` (messages.py:99), `ResponsesRequest` (responses.py:60): add
  `idempotency_key: str | None`; thread it to the resolver.
- `/ws/chat` handler (server.py ws_chat ~`:823`) and the three route handlers:
  pass the key; on an existing id, enter the wait/`_deltas` path for it instead
  of a fresh submit.
- prefix-aware reattach: use `peek` to seed the attaching stream (no change to
  the lock-free contract).
- gates: same-key second submit returns the same id with one row allocated;
  attach after completion returns the stored output; failed key replays the
  error; key reuse after Stop does not resurrect; LRU eviction re-runs; one
  behavioral test per transport shape.
- client (`web/`, after the held hardening PR): generate a UUID per submission,
  include it in the WS ask, reuse on Retry only, drop on Stop; node:test +
  committed-bundle gates.

## Explicitly out of scope

- Exactly-once forever / durable cross-restart dedupe: a server restart clears
  the in-memory registry; a retry after restart re-runs. Durability would need
  persisted keys, which is not justified for a playground.
- Live multi-subscriber token fan-out (two sockets watching one stream
  frame-for-frame); retry gets prefix-replay + tail.
- The external SDKs' own retries: they do not send this field today, so this is
  opt-in and does not change their behaviour.
