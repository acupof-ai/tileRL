# The chat page becomes a build — WebSocket transport, CPU + pending-remote, 2026-09-06

> Status: pending-remote (V100 wall-clock unmeasured; local CPU gates green)

## Context

The playground was `_CHAT_UI`, one 23,898-character Python string in
`ui_assets.py` holding markup, 480 lines of CSS, a hand-written markdown
renderer and a streaming state machine. Two failures shipped from it in three
days — `addThinking` called and never defined, then a page that folded whole
replies into the reasoning block over an empty bubble — because Python only
ever measured that string's length. The gates built afterwards worked around
that: a regex resolver for bare calls, a `node --check` parse pass, a stub-DOM
harness fed hand-written frames.

Replaced with TypeScript under `web/`, built by vite into
`src/tilerl/static/`, served by `StaticFiles`, talking to a new `/ws/chat`
route.

## What Worked

**A WebSocket, because the page both sends and reads.** The old page POSTed a
turn to `/v1/chat/completions` and read the SSE reply — EventSource is
receive-only, so the send and the stream were two different connections with
no shared lifetime. One socket per turn: no request ids on the wire, and a
reload cannot leave a stream attached to the wrong bubble.

**One generator feeds both transports.** `_deltas(request_id, max_new, opened)`
yields `("delta" | "error" | "done", payload, completion_tokens)`; `_stream`
frames those as SSE chunks and `ws_chat` as JSON frames. The lock-free
`engine.peek` poll, the `rstrip("�")` partial-UTF-8 guard, the
`split_think(raw, opened)` split and the partial-closer holdback exist once.
Before this the two routes would each have carried a copy, which is the
mechanism behind #159: `reasoning_content` was wired into the streaming path
only, and a client that flipped `stream` lost it.

**Node construction removes the escaping gate rather than passing it.** The
old renderer built an HTML string for `innerHTML`, so `mdEscape` had to escape
quotes as well as angle brackets — `[x](https://a"onmouseover="alert(1))`
closed the `href` and the rest became a live handler. `render.ts` uses
`createElement`/`createTextNode` only. The 15-input attribute-breakout test is
gone, replaced by a check that no HTML-string sink appears in the bundle or
the sources: the class of bug is absent, not defended.

**Measured, not assumed — the byte deltas:**

| | old (`_CHAT_UI` + `_MD_JS`) | new (`index.html` + bundle) | delta |
|---|---:|---:|---:|
| raw | 28,311 B | 9,038 B | **−68.1%** |
| gzip | 9,814 B | 3,471 B | **−64.6%** |
| Python LOC in `ui_assets.py` | 755 | 97 | −658 |

The frontend sources are 324 lines of TypeScript plus a 94-line `index.html`,
so total authored lines fell from 755 to 418 while gaining a compiler.

**Effect was measured and dropped.** The original plan was effect-ts for the
transport. Built both: Effect's runtime is **182.79 KB** against a **3.07 KB**
stub probe — 60x, and 98.3% of that bundle would have been the library. For a
page whose whole job is one socket and some DOM writes, `try/catch/finally`
around a promise does the same work. ckl ruled it out; the number is why. The
Effect arm was measured before the dependency was removed and is not
reproducible from this tree — reinstall `effect` to re-measure.
`modulePreload: false` took a further **710 B** (5,338 → 4,628) of
MutationObserver polyfill that exists to warm chunks a one-chunk build never
emits.

**The test that would have been green against a 404.**
`TestClient.websocket_connect` fakes the transport in-process, so every WS
assertion here passes with no WebSocket library installed while the deployed
server answers `/ws/chat` with 404. Verified by holding a real `websockets`
client constant and varying only the server's venv: without it,
`InvalidStatus: HTTP 404`; with it, the socket connects. `serve_v100.sh` runs
a plain venv, so the suite carries `find_spec("websockets") is not None` as
its own assertion, and `websockets>=17.1` is in the `server` extra. Deploy
with `uv sync --extra server`.

**The gate then caught a second install path I had not considered, on CI
rather than here.** Both legs went red on that exact assertion while my local
run was green: CI runs `uv sync --dev`, which does **not** install extras, and
the `dev` group duplicates `fastapi`/`httpx`/`uvicorn` precisely because of
that. My local venv had the extra from an earlier `uv build`, so the
dependency was present for a reason unrelated to how it is declared — the
shape of "it works on my machine". `websockets>=17.1` now appears in both the
extra and the `dev` group, and the duplication is the point rather than
redundancy.

Negative control on the fix itself, because an install that is already present
proves nothing: removing the `dev` entry and re-running `uv sync --dev`
uninstalls the package (`- websockets==17.1`) and `find_spec` returns None;
restoring it reinstalls (`+ websockets==17.1`) and the 13 chat-UI tests pass.

**Four negative controls, each fired:**

| mutation | expected failure | observed |
|---|---|---|
| `reasoning_content` → `content` in `_deltas` | phase ordering | 4 tests red, `assert 'planning\n' == ''` on the ordering gate |
| `finish_reason` forced to `"stop"` | truncation notice | `the reasoning stayed folded over an empty reply` — the exact message |
| `replaceChildren(markdown(…))` → `innerHTML =` | sink gate | `assert not ['innerHTML']` |
| `find_spec("websockets")` → None | ws library gate | `uvicorn serves /ws/chat only with a WebSocket protocol implementation installed` |

The second is the one worth keeping: the control failed with the truncation
gate's own message, not an earlier assertion, so it exercises the branch it
names.

## Rule

A test whose transport is faked in-process proves the handler, never the
deploy. Assert the dependency that carries the protocol, or the suite stays
green while the served route 404s — and check every install path that runs the
suite, not only the one on your machine: `uv sync --dev` installs no extras,
so a `server`-extra-only dependency is absent on CI and present locally.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---|---|---|
| 2026-09-06 | pending | Mac (CPU) | cpu | tiny | — | — | n/a — gates only |
| pending-remote | — | V100 | cuda sm70 | Qwen3.8-27B | — | — | — |

No hot-path arithmetic changed: `_deltas` is the same poll loop the SSE route
already ran, moved behind a function boundary. The remote row is the served
tok/s over `/ws/chat` versus the 34.0 tok/s the SSE page last measured, which
needs a card and a redeploy.

Raw artifacts: `web/` sources, `src/tilerl/static/` bundle, byte counts above
reproducible with `wc -c` and `gzip -c`.
