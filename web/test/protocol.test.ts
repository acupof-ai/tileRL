import assert from "node:assert/strict"
import { test } from "node:test"

import { classifyClose, parseFrame } from "../src/protocol.ts"

// The close code cannot say it: a user stop sends 1000 and a finished turn is
// also closed cleanly; the classifier reads the facts that distinguish the
// cases. `opened` separates a mid-turn drop from a handshake that never
// completed (server restarting).
test("a user stop is stopped even if a terminal frame raced it", () => {
  assert.equal(classifyClose(true, false, true), "stopped")
  assert.equal(classifyClose(true, true, true), "stopped")
})

test("a close after a terminal frame is terminal, not dropped", () => {
  assert.equal(classifyClose(false, true, true), "terminal")
})

test("an abnormal close mid-turn is dropped", () => {
  assert.equal(classifyClose(false, false, true), "dropped")
})

test("a close with the socket never opened is unreachable", () => {
  // The supervisor restart window: connection refused, onopen never ran.
  assert.equal(classifyClose(false, false, false), "unreachable")
})

test("a tool_calls frame parses its three string fields", () => {
  const f = parseFrame(JSON.stringify({
    t: "tool_calls",
    tool_calls: [{ id: "call_1_0", type: "function", name: "get_weather",
                  arguments: '{"city":"sf"}' }],
  }))
  assert.ok(f && f.t === "tool_calls")
  if (f?.t !== "tool_calls") throw new Error("narrow")
  assert.equal(f.tool_calls[0]?.name, "get_weather")
  assert.equal(f.tool_calls[0]?.arguments, '{"city":"sf"}')
})

test("a malformed tool_calls frame is null, not a renderable frame", () => {
  assert.equal(parseFrame(JSON.stringify({ t: "tool_calls" })), null)
  assert.equal(parseFrame(JSON.stringify({
    t: "tool_calls", tool_calls: [{ id: 1, name: "x", arguments: "" }],
  })), null)
})

test("a done frame may carry tool_calls", () => {
  const f = parseFrame(JSON.stringify({
    t: "done", finish_reason: "tool_calls",
    tool_calls: [{ id: "c", name: "f", arguments: "{}" }],
    usage: { prompt_tokens: 1, completion_tokens: 1 },
  }))
  assert.ok(f && f.t === "done" && f.tool_calls?.[0]?.name === "f")
})
