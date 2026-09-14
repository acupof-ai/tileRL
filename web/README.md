# Chat web client

TypeScript under `web/`, built to `src/tilerl/static/` (committed) and served by FastAPI.
`pnpm build` = tsc + vite; `pnpm test` = node:test (local-only, no CI Node step; pytest
`test_chat_ui.py` guards the committed artifact).

The source of truth for the frames is the `/ws/chat` route in `server.py`; if this doc and
that route diverge, fix the server or this doc in the same PR.

## Wire contract the UI keys on

`/ws/chat`: the client sends one JSON ask; the server streams JSON text frames, with exactly
one terminal frame last.

- `{"t":"delta","reasoning_content"?:string,"content"?:string}` — 0..N; reasoning frames all
  precede content frames.
- `{"t":"tool_calls","tool_calls":Array<{id,type:"function",name,arguments:string}>}` — 0..1,
  emitted before the terminal frame; 1 entry per call, additive to the contract.
- `{"t":"done","finish_reason":string,"tool_calls"?:Array<…>,"usage":{prompt_tokens,completion_tokens}}`
  — success terminal; `tool_calls` present only when finish_reason is "tool_calls".
- `{"t":"error","message":string}` — failure terminal.

A frame that fails shape validation is dropped (`console.warn`), never fatal.

## Three close reasons (`classifyClose` in protocol.ts)

The socket close code cannot distinguish them — a user stop and a clean finish both end in
1000, and a post-`done` 1001 is normal. The client tracks two booleans: `stopped` (the page
called close) and `terminal` (a `done`/`error` frame arrived).

| stopped | terminal | result   | UI                                                                 |
|---------|----------|----------|--------------------------------------------------------------------|
| true    | *        | stopped  | keep partial reply, no error; joins history if non-empty           |
| false   | true     | terminal | normal finish; settle from the frame                               |
| false   | false    | dropped  | "connection lost … retry?" + Retry; partial reply kept             |

There is no auto-reconnect: neither transport carries frame/request ids or resume — WS has
none, and SSE has no frame-id/resume either — so a resend regenerates the turn and
duplicates every token. Retry is a manual resend of the last user message.

## Server-side contract this assumes (finding 4)

- A mid-stream client hang-up MUST cancel the engine request (GeneratorExit →
  `engine.cancel`); the client reads the close as `dropped` regardless, so an uncancelled
  server run is invisible to the UI.
- A successful turn MUST emit `done` before close, or the UI reports a false drop.
- SSE and WS must keep the same three terminal semantics; the behavioral disconnect gate on
  the server is what keeps this table true.
