import assert from "node:assert/strict"
import { test } from "node:test"

import { pruneTurns } from "../src/render.ts"

// Minimal node: only what pruneTurns reads -- children order, classList, and
// remove() that detaches.
const mkNode = (cls = ""): HTMLElement => {
  const n = {
    _s: new Set(cls.split(" ").filter(Boolean)),
    parent: null as unknown,
    classList: { contains(c: string) { return n._s.has(c) } },
    remove() {
      const p = n.parent as { children: unknown[] } | null
      if (p) p.children.splice(p.children.indexOf(n), 1)
    },
  }
  return n as unknown as HTMLElement
}

const mkLog = (n: number, opts: { pendingIdx?: number[] } = {}): HTMLElement => {
  const log = { children: [] as ReturnType<typeof mkNode>[] }
  for (let i = 0; i < n; i++) {
    const k = mkNode(opts.pendingIdx?.includes(i) ? "pending" : "")
    ;(k as unknown as { parent: unknown }).parent = log
    log.children.push(k)
  }
  return log as unknown as HTMLElement
}

test("pruning removes only the oldest finished turns down to the cap", () => {
  const log = mkLog(43)
  pruneTurns(log, 40)
  assert.equal(log.children.length, 40)
})

test("the in-flight turn is exempt even when older turns fill the cap", () => {
  // 42 nodes, cap 40, original index 1 is pending: the two removals take
  // indexes 0 and 2 and the pending turn survives, shifted up to index 0.
  const log = mkLog(42, { pendingIdx: [1] })
  pruneTurns(log, 40)
  assert.equal(log.children.length, 40)
  assert((log.children[0] as HTMLElement).classList.contains("pending"))
})

test("the last turn is exempt even if it alone would be the overflow", () => {
  // 41 nodes, none pending: 1 removed, last kept (it is at position 40).
  const log = mkLog(41)
  const last = log.children.at(-1)
  pruneTurns(log, 40)
  assert.equal(log.children.length, 40)
  assert.equal(log.children.at(-1), last)
})

test("at or under the cap nothing is removed", () => {
  const log = mkLog(40)
  pruneTurns(log, 40)
  assert.equal(log.children.length, 40)
})
