/** Smooth a token stream into per-frame character reveals.
 *
 * Sparse V100 decode is bursty: graph ticks deliver characters steadily for
 * ~31 frames, then one eager refresh stalls the wire for ~218-272 ms (~13-16
 * frames at 60 Hz) on a strict 32-tick period. The old revealer drained a
 * character every frame whenever ANYTHING was queued and accelerated from a
 * backlog of 8, so its queue was empty almost always: each refresh gap became
 * ~183 ms with no new character on the page (recorded fixture,
 * web/test/fixtures/).
 *
 * The smoother:
 *   1. Banks headroom — holds the first characters HEADROOM_FRAMES before
 *      revealing, so a reserve already exists when the first periodic gap hits.
 *   2. Drains one character per frame while the reserve is healthy, matching
 *      the measured production rate (~24 tok/s × ~2.8 chars/tok ≈ 67 chars/s):
 *      spend the reserve only as production replenishes it.
 *   3. Adds a bounded catch-up term only once a real backlog accumulates
 *      (CATCHUP_BACKLOG), capped at MAX_CHARS, so a regime change or a long
 *      stall never strands characters and never jumps visibly.
 *   4. Flushes everything at once on done/stop/pagehide.
 *
 * Banked by FRAME COUNT, not wall time: animation frames are the 60 Hz clock
 * the page actually reveals on, so no separate time source is needed. Pure
 * logic; the scheduler/canceller are injected, so timing tests replay a
 * recorded arrival timeline on a manual frame stepper with no wall clock.
 */

/** Frames banked before the first reveal. 18 frames ≈ 300 ms at 60 Hz, covering
 * the measured 272 ms refresh stall (≈16 frames) for the first cycles. */
export const HEADROOM_FRAMES = 18
/** Characters revealed per frame while the reserve is healthy. */
const BASE_CHARS = 1
/** Most characters one frame may reveal, however large the backlog. */
export const MAX_CHARS = 8
/** A residual backlog at/over this many characters adds a bounded catch-up term.
 * Kept above the headroom-bank size (~20 chars) so ordinary refresh absorption
 * never accelerates the drain; only a genuine regime change does. */
const CATCHUP_BACKLOG = 24
const CATCHUP_DIVISOR = 8

export interface Reveal {
  /** Queue a freshly arrived chunk; it leaves via onReveal over frames. */
  push: (chunk: string) => void
  /** Reveal every queued character synchronously, once. Terminal/stop/pagehide. */
  flush: () => void
  /** Characters received but not yet revealed. */
  readonly pendingLength: number
}

export interface Timers {
  /** Run cb on the next animation frame. */
  schedule: (cb: () => void) => number
  cancel: (handle: number) => void
}

export const createReveal = (
  onReveal: (chunk: string) => void,
  timers: Timers = {
    schedule: (cb) => requestAnimationFrame(cb),
    cancel: (h) => cancelAnimationFrame(h),
  },
): Reveal => {
  let pending = ""
  let start = 0
  let handle: number | null = null

  /** One-time headroom bank, opened after HEADROOM_FRAMES ticks with content. */
  let banking = true
  let bankFrames = 0

  const queued = (): number => pending.length - start

  const tick = (): void => {
    if (banking) {
      // Still buying headroom: keep the loop alive but reveal nothing.
      if (queued() === 0) {
        handle = null
        return
      }
      bankFrames += 1
      if (bankFrames < HEADROOM_FRAMES) {
        handle = timers.schedule(tick)
        return
      }
      banking = false
    }

    const backlog = queued()
    const catchUp =
      backlog >= CATCHUP_BACKLOG ? Math.floor(backlog / CATCHUP_DIVISOR) : 0
    const n = Math.min(MAX_CHARS, BASE_CHARS + catchUp, backlog)
    if (n > 0) {
      onReveal(pending.slice(start, start + n))
      start += n
    }
    if (queued() === 0) {
      pending = ""
      start = 0
      handle = null
      return
    }
    handle = timers.schedule(tick)
  }

  const kick = (): void => {
    if (handle === null && queued() > 0) handle = timers.schedule(tick)
  }

  return {
    push(chunk: string): void {
      if (chunk === "") return
      if (start > 0 && start === pending.length) {
        pending = ""
        start = 0
      }
      pending += chunk
      kick()
    },
    flush(): void {
      if (handle !== null) {
        timers.cancel(handle)
        handle = null
      }
      const rest = queued() > 0 ? pending.slice(start) : ""
      pending = ""
      start = 0
      banking = true
      bankFrames = 0
      if (rest !== "") onReveal(rest)
    },
    get pendingLength(): number {
      return queued()
    },
  }
}
