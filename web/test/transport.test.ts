import assert from "node:assert/strict"
import { test } from "node:test"

import type { Frame } from "../src/protocol.ts"
import { ask, waitForHealth } from "../src/transport.ts"

// A scripted socket exposed to the test: the test pushes frames and fires the
// close itself, so each real ending is driven through `ask`, not reconstructed.
// `autoOpen` distinguishes a healthy socket (onopen runs) from a refused
// handshake (the restart window: onerror/onclose with no onopen).
class FakeSocket {
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  sent: string[] = []
  url: string
  constructor(url: string, autoOpen = true) {
    this.url = url
    if (autoOpen) queueMicrotask(() => this.onopen?.())
  }
  send(d: string) {
    this.sent.push(d)
  }
  failHandshake() {
    queueMicrotask(() => {
      this.onerror?.()
      this.onclose?.()
    })
  }
  close() {
    queueMicrotask(() => this.onclose?.())
  }
}

let current: FakeSocket
let autoOpen = true
;(globalThis as { WebSocket: unknown }).WebSocket = class {
  constructor(url: string) {
    current = new FakeSocket(url, autoOpen)
    return current
  }
}

const drive = async (
  mode: "stop" | "terminal" | "dropped" | "unreachable",
  frames: string[],
): Promise<{ kind: string; seen: Frame[] }> => {
  autoOpen = mode !== "unreachable"
  const seen: Frame[] = []
  let stop: (() => void) | null = null
  const p = ask(
    "ws://x/ws/chat",
    { messages: [] },
    (f) => {
      seen.push(f)
      if (mode === "stop" && f.t === "delta") stop?.()
    },
    (s) => {
      stop = s
    },
  )
  await new Promise((r) => setTimeout(r, 0))
  if (mode === "unreachable") {
    current.failHandshake()
  } else {
    for (const d of frames) current.onmessage?.({ data: d })
    await new Promise((r) => setTimeout(r, 0))
    if (mode !== "stop") current.onclose?.()
  }
  const kind = await p
  return { kind, seen }
}

test("a user stop resolves stopped, even with frames in flight", async () => {
  const { kind } = await drive("stop", [
    JSON.stringify({ t: "delta", content: "partial" }),
    JSON.stringify({ t: "delta", content: " more" }),
  ])
  assert.equal(kind, "stopped")
})

test("a close after a done frame resolves terminal", async () => {
  const { kind } = await drive("terminal", [
    JSON.stringify({ t: "delta", content: "hi" }),
    JSON.stringify({ t: "done", finish_reason: "stop",
      usage: { prompt_tokens: 1, completion_tokens: 1 } }),
  ])
  assert.equal(kind, "terminal")
})

test("a server restart mid-turn resolves dropped, not stopped", async () => {
  // No terminal frame and no user stop: the restart case that must not be
  // mislabelled as a deliberate stop.
  const { kind } = await drive("dropped", [JSON.stringify({ t: "delta", content: "half" })])
  assert.equal(kind, "dropped")
})

test("a refused handshake resolves unreachable", async () => {
  // Supervisor restart: connection refused, onopen never ran. This must not be
  // reported as dropped (the caller polls /health instead of offering retry).
  const { kind } = await drive("unreachable", [])
  assert.equal(kind, "unreachable")
})

test("an unparseable frame is dropped without ending the stream", async () => {
  autoOpen = true
  const seen: Frame[] = []
  const p = ask("ws://x/ws/chat", {}, (f) => seen.push(f))
  await new Promise((r) => setTimeout(r, 0))
  const origWarn = console.warn
  console.warn = () => {}
  current.onmessage?.({ data: "{not json" })
  current.onmessage?.({ data: JSON.stringify({ t: "delta", content: "ok" }) })
  current.onclose?.()
  const kind = await p
  console.warn = origWarn
  assert.equal(kind, "dropped", "no terminal frame arrived, so the close is dropped")
  assert.equal(seen.length, 1)
  assert.equal((seen[0] as { content?: string }).content, "ok")
})

// waitForHealth with scripted fetch and virtual timers.
const installFetch = (responses: Array<{ ok: boolean } | "throw">): void => {
  let i = 0
  ;(globalThis as { fetch: unknown }).fetch = async () => {
    const r = responses[i] ?? responses.at(-1)
    i += 1
    if (r === "throw") throw new Error("refused")
    return { ok: (r as { ok: boolean }).ok }
  }
}

test("waitForHealth resolves true once /health answers 200", async () => {
  installFetch(["throw", { ok: false }, { ok: true }])
  const ticks: number[] = []
  const ok = await waitForHealth(
    "/health",
    (a) => ticks.push(a),
    { sleep: async () => {}, now: () => 0 },
  )
  assert.equal(ok, true)
  assert.deepEqual(ticks, [1, 2])
})

test("waitForHealth resolves false past the timeout", async () => {
  installFetch([{ ok: false }])
  let t = 0
  const ok = await waitForHealth(
    "/health",
    () => {},
    { sleep: async (ms) => { t += ms }, now: () => t },
  )
  assert.equal(ok, false)
})

test("waitForHealth aborts when the abort function is invoked", async () => {
  installFetch([{ ok: false }])
  let abortFn: (() => void) | null = null
  const p = waitForHealth(
    "/health",
    () => {},
    { sleep: async () => {}, now: () => 0 },
    (a) => { abortFn = a },
  )
  abortFn?.()
  assert.equal(await p, false)
})
