# The chat page becomes a build — WebSocket transport, V100 + CPU, 2026-09-06

> Status: Shipped (#175, `363de2a`; served on the V100, throughput below)

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
restoring it reinstalls (`+ websockets==17.1`) and the chat-UI tests pass.

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

**The stop cut folded into `_deltas` on the rebase, and my first control was
aimed at the wrong files.** #174 landed the stop sequences with its cut inside
`_stream`; rebasing onto it put the cut in the shared generator instead, so
`/ws/chat` gets it without a second copy. 48's finding is that
`_ScriptedEngine` honours `stop_texts` itself, so a route arm stays green with
the *engine's* matching deleted — all six of theirs did. Two arms therefore,
one per layer.

I deleted the streaming cut and ran `test_server.py` + `test_chat_ui.py`: **48
passed**. That reads as "the fold is untested" and was instead my control
pointed at files that never exercised it — the gate lives in
`test_api_sdk.py`. Re-run there, both halves are load-bearing separately:

| revert | leaked text | expected |
|---|---|---|
| whole cut deleted | `The answer is` | `The answer` |
| holdback only, no completed-match cut | `The answer ` | `The answer` |

The second reproduces 48's own first error exactly — the trailing space of
`" is"`, in the frame before the last. The arithmetic, from 48: the stop is
three characters, so `hold` is 2 and the frame carries everything up to
`len(text) - 2`, which is one character past the match's start. That is why a
holdback fails on a *completed* match rather than on a forming one: it is
sized for the prefix case and a completed match is already inside the window
it releases.

The WS arm added here (`test_the_websocket_route_never_emits_a_stop_sequence`)
is red under both reverts, failing on its own assertion rather than an earlier
one, so it covers the layer the two transports share. The route also had to
learn `stop`: the page sends none, but a gate on a field the route silently
drops would pass on an empty list forever.

**What is NOT gated here, and why it is not on this double.** v100 suggested
forcing the cut mid-token, where `len(output_ids)` and the decoded-character
count diverge most sharply. `_ByteTokenizer` cannot express it: decoding every
prefix of `"The answer is 4."` gives 17 tokens for 16 characters — k=1 → `''`,
then one character per step to k=17. After a single leading no-character
token, every token carries exactly one character, so a stop string always
begins at a token boundary and the two counts stay locked. `" is"` as one
token is a real-tokenizer property this double structurally lacks, so that arm
belongs in `test_e2e.py` against the real tokenizer, which is where 48 put the
genuine stop gate for the same reason.

**Two things review found that are not fixed here, recorded so the next reader
does not rediscover them.** From 48: if a stream dies after a stop matched,
`_deltas`'s error branch returns before `engine.stop_text(request_id)` runs, so
the `_finished_stop` entry never pops and outlives the request — bounded to
one short string per abandoned stream that actually matched, and being fixed
engine-side rather than by asking every caller to remember. `gen.close()` on
`WebSocketDisconnect` is the same shape and equally bounded. Second: the WS
route's `except Exception` turns a `_submit` refusal (`tool_choice`,
`stop_texts` without a decode) into an `{"t": "error"}` frame, so the page
cannot tell "refused" from "the engine died". Correct for a socket, which has
no status code, and the place to change if the playground should ever show a
refusal reason.

**A committed artifact can go stale and nothing here catches it.** The bundle
is built from `web/src/` and committed, so an edit to a source without
`npm run build` leaves a bundle that is internally consistent and simply old:
every test passes and the served page is the previous version. That is the
`_CHAT_UI` defect reintroduced one layer out, where the string Python only
measured the length of becomes a bundle nothing rebuilds — and removing that
defect is this PR's whole justification.

**The id gate is not a partial version of this check.** It catches an id
renamed in `index.html` without a rebuild; this is a changed `render.ts` with
no rebuild. Different failure, different symptom, zero overlap — a reader who
sees "there is a gate on the bundle" will assume coverage that does not exist.

Verified by hand for this PR (rebuild, `diff -r`, identical) and checked that
nothing ahead in the queue touches those paths: #177, #178 and
`753da30..origin/main` all touch neither `web/` nor `src/tilerl/static/`,
confirmed independently by 48.

Not gated in CI here because it needs node, and this tree's convention is that
node-dependent gates skip where node is absent — which would make it a gate
that does not run on the pod. Two upgrade paths, cheapest first: commit a hash
of `web/src/` beside the bundle and assert in Python that it matches a rehash
(no node, runs everywhere, catches exactly "sources changed, bundle didn't");
or a CI step that rebuilds and diffs, once node is a hard CI dependency.

## Rule

A test whose transport is faked in-process proves the handler, never the
deploy. Assert the dependency that carries the protocol, or the suite stays
green while the served route 404s — and check every install path that runs the
suite, not only the one on your machine: `uv sync --dev` installs no extras,
so a `server`-extra-only dependency is absent on CI and present locally.

## Results

| date | commit | machine | target | model | ttft s | throughput tok/s | tokens |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-06 | 363de2a | Mac (CPU) | cpu | tiny | — | n/a — gates only | — |
| 2026-09-06 17:5x +0800 | 363de2a | V100 | cuda sm70 | Qwen3.8-27B | 0.57 | **46.5** | 1500 (cap, `length`) |
| 2026-09-06 17:5x +0800 | 363de2a | V100 | cuda sm70 | Qwen3.8-27B | 0.78 | **50.4** | 491 (`stop`) |

Measured by 27 over `/ws/chat`, idle endpoint, thinking on, cap 1500,
**client-side timing from the first delta to the last, one request per row** —
not a server-side counter and not an average over repeats, so treat these as
single-sample. Row 1 is a 300-word essay in 32.3 s over 817 frames; row 2 an
HTML page in 9.7 s over 254 frames.

**The coalescing ratio holds on the real model: 1.84 and 1.93 tokens per
frame**, against the 1.83 the SSE path measured on the 27B and recorded in
`_stream`'s own comment. That resolves the one anomaly from the first live
run, where a 74-character reasoning block arrived in 21 frames (3.5
chars/frame) and looked like a different coalescing regime; it was the
short-block case, where the 20 ms poll dominates rather than the decode.

Row 1's `length` finish is the truncation branch firing on real weights —
until this run it was exercised only on the CPU double.

**Against the 34.0 tok/s the SSE page last measured: both figures stand, no
ratio.** Different day, different card state, different measurement harness;
dividing them would manufacture a speedup out of two unpaired samples. A
transport comparison needs both arms on one card in one session, which nobody
has run.

No hot-path arithmetic changed: `_deltas` is the same poll loop the SSE route
already ran, moved behind a function boundary.

Raw artifacts: `web/` sources, `src/tilerl/static/` bundle, byte counts above
reproducible with `wc -c` and `gzip -c`.
