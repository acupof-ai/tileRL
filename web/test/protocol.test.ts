import assert from "node:assert/strict"
import { test } from "node:test"

import { classifyClose } from "../src/protocol.ts"

// The close code cannot say it: a user stop sends 1000 and a finished turn is
// also closed cleanly; the classifier reads the two facts that distinguish the
// three cases.
test("a user stop is stopped even if a terminal frame raced it", () => {
  assert.equal(classifyClose(true, false), "stopped")
  assert.equal(classifyClose(true, true), "stopped")
})

test("a close after a terminal frame is terminal, not dropped", () => {
  // 1001 from a server restarting right after `done`: the reply is complete.
  assert.equal(classifyClose(false, true), "terminal")
})

test("an abnormal close mid-turn is dropped", () => {
  assert.equal(classifyClose(false, false), "dropped")
})
