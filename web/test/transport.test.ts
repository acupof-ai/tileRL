import assert from "node:assert/strict"
import { test } from "node:test"

import type { Frame } from "../src/protocol.ts"
import { ask } from "../src/transport.ts"

// A scripted socket exposed to the test: the test pushes frames and fires the
// close itself, so each real ending is driven through `ask`, not reconstructed.
class FakeSocket {
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  sent: string[] = []
  url: string
  constructor(url: string) {
    this.url = url
    queueMicrotask(() => this.onopen?.())
  }
  send(d: string) {
    this.sent.push(d)
  }
  close() {
    queueMicrotask(() => this.onclose?.())
  }
}

let current: FakeSocket
;(globalThis as { WebSocket: unknown }).WebSocket = class {
  constructor(url: string) {
    current = new FakeSocket(url)
    return current
  }
}

const drive = async (
  mode: "stop" | "terminal" | "dropped",
  frames: string[],
): Promise<{ kind: string; seen: Frame[] }> => {
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
  for (const d of frames) current.onmessage?.({ data: d })
  await new Promise((r) => setTimeout(r, 0))
  if (mode !== "stop") current.onclose?.()
  const kind = await p
  return { kind, seen }
}

test("a user stop resolves stopped, even with frames in flight", async () => {
  const { kind, seen } = await drive("stop", [
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

test("an unparseable frame is dropped without ending the stream", async () => {
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
