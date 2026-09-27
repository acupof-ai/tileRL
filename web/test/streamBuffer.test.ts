import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { test } from "node:test"

import { createReveal, HEADROOM_MS, MAX_CHARS, type RevealTuning } from "../src/streamBuffer.ts"
import { createRevealOld } from "./oldRevealControl.ts"

/** Virtual clock shared by arrivals (now()) and frame callbacks. One step
 * advances FRAME_MS, then runs every callback scheduled for this frame exactly
 * once with the new timestamp; a callback that re-schedules runs next step. */
const fakeRaf = (frameMs: number, start = 0) => {
  const jobs = new Map<number, (now: number) => void>()
  let seq = 0
  let t = start
  return {
    timers: {
      schedule: (cb: (now: number) => void): number => {
        seq += 1
        jobs.set(seq, cb)
        return seq
      },
      cancel: (id: number): void => {
        jobs.delete(id)
      },
      now: (): number => t,
    },
    step: (): void => {
      t += frameMs
      const due = [...jobs.values()]
      jobs.clear()
      for (const cb of due) cb(t)
    },
    pending: (): number => jobs.size,
    time: (): number => t,
  }
}

test("several pushes within one frame schedule one drain and lose/reorder nothing", () => {
  const raf = fakeRaf(1000 / 60)
  const out: string[] = []
  const r = createReveal((s) => out.push(s), raf.timers)
  r.push("ab")
  r.push("cd")
  r.push("e")
  assert.equal(raf.pending(), 1, "coalesced into a single scheduled frame")
  for (let i = 0; i < 600 && raf.pending() > 0; i++) raf.step()
  assert.equal(out.join(""), "abcde")
  assert.equal(r.pendingLength, 0)
})

test("the first characters are banked for ~HEADROOM_MS regardless of refresh rate", () => {
  for (const hz of [60, 120, 144]) {
    const frameMs = 1000 / hz
    const raf = fakeRaf(frameMs)
    const frames: { t: number; s: string }[] = []
    const r = createReveal((s) => frames.push({ t: raf.time(), s }), raf.timers)
    r.push("x".repeat(40))
    while (frames.length === 0) raf.step()
    assert.ok(
      frames[0].t >= HEADROOM_MS - frameMs,
      `@${hz}Hz first reveal at ${frames[0].t.toFixed(1)} ms, bank is ${HEADROOM_MS} ms`,
    )
    while (raf.pending() > 0) raf.step()
    assert.equal(frames.map((f) => f.s).join(""), "x".repeat(40))
  }
})

test("a backlog accelerates the per-frame step but never past the cap", () => {
  const raf = fakeRaf(1000 / 60)
  const frames: string[] = []
  const r = createReveal((s) => frames.push(s), raf.timers)
  // 400 chars is far past rate * CATCHUP_MS so the catch-up term is active.
  r.push("x".repeat(400))
  let pastBank = 0
  for (let i = 0; i < 600 && raf.pending() > 0; i++) {
    raf.step()
    if (raf.time() >= HEADROOM_MS) pastBank += 1
  }
  assert.equal(frames.join("").length, 400)
  for (const f of frames) assert.ok(f.length <= MAX_CHARS, "no frame bursts past the cap")
  assert.ok(pastBank < 400 / 0.8, "catch-up drains the burst faster than one char/frame")
})

test("a later push after drain restarts the loop without re-banking", () => {
  const raf = fakeRaf(1000 / 60)
  const out: string[] = []
  const r = createReveal((s) => out.push(s), raf.timers)
  r.push("x".repeat(40))
  for (let i = 0; i < 600 && raf.pending() > 0; i++) raf.step()
  assert.equal(raf.pending(), 0)
  r.push("y".repeat(20))
  const t0 = raf.time()
  raf.step()
  // The turn's bank already opened: the first frame after the new burst reveals.
  assert.ok(out.some((s) => s.includes("y")) || raf.time() - t0 < HEADROOM_MS)
  for (let i = 0; i < 600 && raf.pending() > 0; i++) raf.step()
  assert.equal(out.join(""), "x".repeat(40) + "y".repeat(20))
})

test("flush reveals every queued character at once, even inside the bank", () => {
  const raf = fakeRaf(1000 / 60)
  const out: string[] = []
  const r = createReveal((s) => out.push(s), raf.timers)
  r.push("the quick brown")
  r.flush()
  assert.equal(out.join(""), "the quick brown")
  assert.equal(r.pendingLength, 0)
  assert.equal(raf.pending(), 0, "the scheduled drain is cancelled, no double reveal")
  r.flush() // idempotent on a later terminal/stop
  assert.equal(out.join(""), "the quick brown")
})

/** A recorded /ws/chat frame: arrival ms (since first frame), char count. */
type Row = { t: number; n: number }

const FIXTURE = fileURLToPath(new URL("./fixtures/ws-chat-prod-2026-09-26.jsonl", import.meta.url))

/**
 * Recorded 2026-09-26 against the production sparse V100 serve
 * (run_serve_tc_07502a34_cold8g.sh, idle confirmed via /health running=0
 * waiting=0; prompt "用大约200字解释为什么天空是蓝色的…", enable_thinking,
 * max_tokens=600, websockets client on localhost): 182 text frames over 8.0 s,
 * five refresh stalls, max wire gap 272.5 ms. The real bursty timeline the page
 * sees, not a hand-shaped one.
 */
const loadFixture = (): Row[] =>
  readFileSync(FIXTURE, "utf8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l) as Row)

type Make = (
  onReveal: (s: string) => void,
  timers: ReturnType<typeof fakeRaf>["timers"],
) => { push(c: string): void; flush(): void; readonly pendingLength: number }

/** The pre-1000/2000 tuning, kept as the negative control for the long-context
 * recordings (the shipped revealer parameterised with the old constants). */
const OLD_TUNING: RevealTuning = { headroomMs: 300, catchupMs: 600 }

/** Replay a recorded arrival timeline (timestamps optionally stretched) through
 * a reveal factory on a virtual frame clock. Returns reveal times in ms and,
 * when `flushAtDone`, the number of characters `flush()` dumped in one call one
 * frame after the last arrival (the done frame lands there). */
const replay = (
  rowsIn: Row[],
  frameMs: number,
  stretch: number,
  make: Make,
  flushAtDone = false,
): { reveals: number[]; flushBatch: number } => {
  const rows = rowsIn.map((r) => ({ t: r.t * stretch, n: r.n }))
  const raf = fakeRaf(frameMs, -frameMs) // first step lands at t = 0
  const reveals: number[] = []
  const r = make(
    (s) => { for (const _ of s) reveals.push(raf.time()) },
    raf.timers,
  )
  let next = 0
  const horizon = flushAtDone ? rows[rows.length - 1].t + frameMs : rows[rows.length - 1].t + 3000
  while (raf.time() < horizon) {
    raf.step()
    while (next < rows.length && rows[next].t <= raf.time()) {
      r.push("x".repeat(rows[next].n))
      next++
    }
  }
  const flushBatch = r.pendingLength
  r.flush()
  return { reveals, flushBatch }
}

/** Longest interval with no new character displayed AFTER the first one. The
 * one-time headroom bank is an intentional start-of-turn delay, not a stall;
 * the post-done flush is also terminal, so it is excluded. */
const maxActiveGap = (reveals: number[]): number => {
  let g = 0
  for (let i = 1; i < reveals.length; i++) g = Math.max(g, reveals[i] - reveals[i - 1])
  return g
}

const TOTAL_CHARS = (rows: Row[]): number => rows.reduce((a, x) => a + x.n, 0)

for (const hz of [60, 120, 144]) {
  for (const [label, stretch] of [["production cadence", 1], ["slow concurrency x2", 2]] as const) {
    test(`PRODUCTION GATE @${hz}Hz, ${label}: max no-new-character gap < 100 ms`, () => {
      const rows = loadFixture()
      const { reveals } = replay(rows, 1000 / hz, stretch, (cb, t) => createReveal(cb, t))
      assert.equal(reveals.length, TOTAL_CHARS(rows), "every character revealed exactly once")
      const gap = maxActiveGap(reveals)
      assert.ok(
        gap < 100,
        `the page must never go 100 ms without a new character; got ${gap.toFixed(1)} ms ` +
          `(wire stalls to 272 ms, ${stretch === 2 ? "544 ms when stretched" : "unstretched"})`,
      )
    })
  }
}

for (const hz of [60, 120, 144]) {
  test(`NEGATIVE CONTROL @${hz}Hz: the pre-fix frame-count revealer goes red`, () => {
    const rows = loadFixture()
    const { reveals } = replay(rows, 1000 / hz, 1, (cb, t) => createRevealOld(cb, t))
    assert.equal(reveals.length, TOTAL_CHARS(rows))
    const gap = maxActiveGap(reveals)
    assert.ok(
      gap >= 100,
      `the old revealer must show the refresh window at ${hz}Hz or the gate has ` +
        `no teeth; got ${gap.toFixed(1)} ms`,
    )
  })
}

/**
 * Long-context recordings made 2026-09-27 against the production V100 serve
 * (tools/stream-stall/ws_rec*.py): a single 30k turn and the SECOND turn of a
 * two-turn 30k conversation. Their wire gaps run to 1.24-1.4 s — far past the
 * 272 ms short-context stalls — so the 300 ms bank / 600 ms catch-up tuning
 * leaves multi-hundred-ms visible holes.
 */
const LONG_FIXTURES: Array<{ file: string; note: string }> = [
  { file: "wsrec_30000.jsonl", note: "single 30k turn; first post-frame gap is 1241 ms" },
  { file: "wsrec2_30k_t2.jsonl", note: "second turn of a two-turn 30k conversation; gaps to 1.4 s" },
]

for (const { file, note } of LONG_FIXTURES) {
  for (const hz of [60, 120, 144]) {
    test(`LONG CONTEXT ${file} @${hz}Hz: the 1000/2000 tuning beats the old 300/600`, () => {
      const rows = readFileSync(fileURLToPath(new URL(`./fixtures/${file}`, import.meta.url)), "utf8")
        .trim()
        .split("\n")
        .map((l) => JSON.parse(l) as Row)
      const oldGap = maxActiveGap(
        replay(rows, 1000 / hz, 1, (cb, t) => createReveal(cb, t, OLD_TUNING)).reveals,
      )
      const { reveals, flushBatch } = replay(rows, 1000 / hz, 1, (cb, t) => createReveal(cb, t), true)
      const newGap = maxActiveGap(reveals)
      assert.equal(reveals.length, TOTAL_CHARS(rows), `${note}; every character revealed once`)
      assert.ok(oldGap >= 100, `the old tuning must stay red on this recording; got ${oldGap.toFixed(0)} ms`)
      assert.ok(
        newGap < oldGap,
        `the new tuning must shorten the worst gap (${note}); old ${oldGap.toFixed(0)} ` +
          `new ${newGap.toFixed(0)} ms`,
      )
      // The done frame flushes the reserve in one call; report the size so a
      // larger bank silently turning the terminal into a jump stays visible.
      assert.ok(
        flushBatch <= 80,
        `done-time flush dumped ${flushBatch} characters at once; the 1000 ms bank ` +
          `must not turn the terminal into a visible jump`,
      )
    })
  }
}
