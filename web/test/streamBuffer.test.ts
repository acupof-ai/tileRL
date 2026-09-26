import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { test } from "node:test"

import { createReveal, HEADROOM_FRAMES, MAX_CHARS } from "../src/streamBuffer.ts"
import { createRevealOld } from "./oldRevealControl.ts"

/** One frame = run every callback scheduled at frame start, exactly once. A
 * callback that re-schedules (the drain loop) runs on the next step. */
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
  // Past the headroom bank; drain fully until the queue settles.
  for (let i = 0; i < 600 && raf.pending() > 0; i++) raf.step()
  assert.equal(out.join(""), "abcde")
  assert.equal(r.pendingLength, 0)
})

test(`the first characters are banked for HEADROOM_FRAMES (${HEADROOM_FRAMES}), then revealed one per frame`, () => {
  const raf = fakeRaf()
  const frames: string[] = []
  const r = createReveal((s) => frames.push(s), raf.timers)
  r.push("abcdefgh")
  for (let i = 0; i < HEADROOM_FRAMES - 1; i++) raf.step()
  assert.equal(frames.length, 0, "nothing is revealed inside the headroom bank")
  raf.step() // frame HEADROOM_FRAMES: the bank opens
  assert.deepEqual(frames, ["a"], "the bank opens with one character")
  raf.step()
  raf.step()
  assert.deepEqual(frames.slice(1), ["b", "c"], "one char/frame once open")
  while (raf.pending() > 0) raf.step()
  assert.equal(frames.join(""), "abcdefgh")
})

test("a backlog accelerates the per-frame step but never past the cap", () => {
  const raf = fakeRaf()
  const frames: string[] = []
  const r = createReveal((s) => frames.push(s), raf.timers)
  r.push("x".repeat(56))
  for (let i = 0; i < HEADROOM_FRAMES; i++) raf.step() // pass the bank
  assert.equal(frames[0]?.length, MAX_CHARS, "56-char backlog -> floor(56/8)=7 + 1, capped")
  while (raf.pending() > 0) raf.step()
  assert.equal(frames.join("").length, 56)
  for (const f of frames) assert.ok(f.length <= MAX_CHARS, "no frame bursts past the cap")
})

test("a later push after drain restarts the loop without re-banking", () => {
  const raf = fakeRaf()
  const out: string[] = []
  const r = createReveal((s) => out.push(s), raf.timers)
  r.push("abcdefgh")
  for (let i = 0; i < 30; i++) raf.step()
  assert.equal(raf.pending(), 0)
  r.push("i")
  raf.step() // the turn's bank was already opened: reveal immediately
  assert.equal(out[out.length - 1], "i")
})

test("flush reveals every queued character at once and empties the queue", () => {
  const raf = fakeRaf()
  const out: string[] = []
  const r = createReveal((s) => out.push(s), raf.timers)
  r.push("the quick brown")
  r.flush() // even inside the bank
  assert.equal(out.join(""), "the quick brown")
  assert.equal(r.pendingLength, 0)
  assert.equal(raf.pending(), 0, "the scheduled drain is cancelled, no double reveal")
  // Idempotent on a later terminal/stop.
  r.flush()
  assert.equal(out.join(""), "the quick brown")
})

/** A recorded /ws/chat frame: arrival ms (since first frame), char count. */
type Row = { t: number; n: number }

const FIXTURE = fileURLToPath(new URL("./fixtures/ws-chat-prod-2026-09-26.jsonl", import.meta.url))
const FRAME_MS = 1000 / 60

/**
 * Recorded 2026-09-26 against the production sparse V100 serve
 * (run_serve_tc_07502a34_cold8g.sh, idle confirmed via /health running=0
 * waiting=0; prompt "用大约200字解释为什么天空是蓝色的…", enable_thinking,
 * max_tokens=600, websockets client on localhost): 182 text frames over 8.0 s,
 * five refresh stalls, max wire gap 272.5 ms. This is the real bursty timeline
 * the page sees, not a hand-shaped one.
 */
const loadFixture = (): Row[] =>
  readFileSync(FIXTURE, "utf8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l) as Row)

/** Replay a recorded arrival timeline through a reveal factory on a virtual
 * 60 Hz frame clock; returns the virtual time (ms) of every revealed char. */
const replay = (
  rows: Row[],
  make: (onReveal: (s: string) => void, timers: { schedule: (cb: () => void) => number; cancel: (h: number) => void }) => { push(c: string): void; flush(): void },
): { reveals: number[]; total: number } => {
  const jobs = new Map<number, () => void>()
  let seq = 0
  let vt = -FRAME_MS // the first step lands at t = 0, when frame 0 arrives
  const reveals: number[] = []
  const r = make(
    (s) => { for (const _ of s) reveals.push(vt) },
    {
      schedule: (cb) => { seq += 1; jobs.set(seq, cb); return seq },
      cancel: (h) => jobs.delete(h),
    },
  )
  let next = 0
  const horizon = rows[rows.length - 1].t + 5000
  while (vt < horizon) {
    vt += FRAME_MS
    while (next < rows.length && rows[next].t <= vt) {
      r.push("x".repeat(rows[next].n))
      next++
    }
    const due = [...jobs.values()]
    jobs.clear()
    for (const cb of due) cb()
  }
  r.flush()
  return { reveals, total: rows.reduce((a, x) => a + x.n, 0) }
}

/** Longest interval with no new character displayed AFTER the first one (the
 * one-time headroom bank is an intentional start-of-turn delay, not a stall). */
const maxActiveGap = (reveals: number[]): number => {
  let g = 0
  for (let i = 1; i < reveals.length; i++) g = Math.max(g, reveals[i] - reveals[i - 1])
  return g
}

test("PRODUCTION GATE: max no-new-character gap is under 100 ms on the recorded timeline", () => {
  const rows = loadFixture()
  const { reveals, total } = replay(rows, (cb, t) => createReveal(cb, t))
  assert.equal(reveals.length, total, "every produced character is revealed exactly once")
  const gap = maxActiveGap(reveals)
  assert.ok(
    gap < 100,
    `while the message streams, the page must never go 100 ms without a new ` +
      `character; recorded max gap ${gap.toFixed(1)} ms (wire stalls to 272 ms)`,
  )
})

test("NEGATIVE CONTROL: the pre-fix revealer goes red on the same timeline", () => {
  const rows = loadFixture()
  const { reveals, total } = replay(rows, (cb, t) => createRevealOld(cb, t))
  assert.equal(reveals.length, total)
  const gap = maxActiveGap(reveals)
  assert.ok(
    gap >= 100,
    `the old drain-1-per-frame-with-no-bank buffer must exhibit the ~218 ms ` +
      `refresh window on this fixture, or the gate has no teeth; got ${gap.toFixed(1)} ms`,
  )
})
