import assert from "node:assert/strict"
import { test } from "node:test"

import { createReveal, MAX_CHARS } from "../src/streamBuffer.ts"

/** One frame = run every callback that was scheduled at frame start, exactly
 * once. A callback that re-schedules (the drain loop) runs on the next step. */
const fakeRaf = () => {
  const jobs = new Map<number, () => void>()
  let seq = 0
  return {
    timers: {
      schedule: (cb: () => void): number => {
        seq += 1
        jobs.set(seq, cb)
        return seq
      },
      cancel: (id: number): void => {
        jobs.delete(id)
      },
    },
    step: (): void => {
      const due = [...jobs.values()]
      jobs.clear()
      for (const cb of due) cb()
    },
    pending: (): number => jobs.size,
  }
}

test("several pushes within one frame schedule one drain and lose/reorder nothing", () => {
  const raf = fakeRaf()
  const out: string[] = []
  const r = createReveal((s) => out.push(s), raf.timers)
  r.push("ab")
  r.push("cd")
  r.push("e")
  assert.equal(raf.pending(), 1, "coalesced into a single scheduled frame")
  // Drain fully: repeatedly run frames until the queue settles.
  for (let i = 0; i < 10 && raf.pending() > 0; i++) raf.step()
  assert.equal(out.join(""), "abcde")
  assert.equal(r.pendingLength, 0)
})

test("a thin stream reveals a steady one character per frame", () => {
  const raf = fakeRaf()
  const frames: string[] = []
  const r = createReveal((s) => frames.push(s), raf.timers)
  r.push("abc")
  raf.step()
  raf.step()
  raf.step()
  assert.deepEqual(frames, ["a", "b", "c"])
  assert.equal(raf.pending(), 0, "idle after the queue drains")
})

test("a backlog accelerates the per-frame step but never past the cap", () => {
  const raf = fakeRaf()
  const frames: string[] = []
  const r = createReveal((s) => frames.push(s), raf.timers)
  r.push("x".repeat(56))
  raf.step() // queued 56 -> floor(56/8)=7 -> capped MAX_CHARS
  assert.equal(frames[0]?.length, MAX_CHARS)
  while (raf.pending() > 0) raf.step()
  assert.equal(frames.join("").length, 56)
  for (const f of frames) assert.ok(f.length <= MAX_CHARS, "no frame bursts past the cap")
})

test("a later push after drain restarts the loop", () => {
  const raf = fakeRaf()
  const out: string[] = []
  const r = createReveal((s) => out.push(s), raf.timers)
  r.push("a")
  raf.step()
  assert.equal(raf.pending(), 0)
  r.push("bc")
  assert.equal(raf.pending(), 1)
  while (raf.pending() > 0) raf.step()
  assert.equal(out.join(""), "abc")
})

test("flush reveals every queued character at once and empties the queue", () => {
  const raf = fakeRaf()
  const out: string[] = []
  const r = createReveal((s) => out.push(s), raf.timers)
  r.push("the quick brown")
  r.flush()
  assert.equal(out.join(""), "the quick brown")
  assert.equal(r.pendingLength, 0)
  assert.equal(raf.pending(), 0, "the scheduled drain is cancelled, no double reveal")
  // Idempotent on a later terminal/stop.
  r.flush()
  assert.equal(out.join(""), "the quick brown")
})
